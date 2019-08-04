import os, argparse
import torch
from torchvision import models
import torch.nn as nn
from scipy import signal
import logging
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import roc_auc_score

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

from train_utils import AgeDataHandler, init_visdom
from base_utils import Params, set_logger, parse_soundfile

def compute_loss(batch, backward = False):
    observations = []
    targets = torch.zeros(len(batch))
    for i, (soundfile, category) in enumerate(batch):
        Sxx = parse_soundfile(soundfile, timeframe, window_fn, features)
        observations.append(Sxx)
        targets[i] = 2.0*(category < 6) - 1 if category is not None else 0

    observations = torch.stack(observations)
    if cuda:
        observations = observations.cuda()
        targets = targets.cuda()

    outputs = model(observations)
    dist = torch.sum((outputs - _C_) ** 2, dim=1)
    losses  = torch.where(targets == 0, dist, ETA * ((dist + eps) ** targets.float()) )
    loss = torch.mean(losses)
    if backward:
        loss.backward()
        return loss.detach().item() / len(batch)
    else:
        return loss.detach().item() / len(batch), dist.cpu().data.numpy().tolist(), targets.cpu().data.numpy().tolist()

def eval_model(val_data):
    total_loss, total_obs = 0, 0
    scores, targets = [], []
    for i, batch in enumerate(val_data):
        batch_loss, x,y = compute_loss(batch, backward = False)
        total_loss +=  batch_loss * len(batch)
        total_obs += len(batch)
        scores += x
        targets += y

    auc = roc_auc_score(targets, scores)
    return total_loss/total_obs, auc

def train(model, train_data, optimizer):
    best_auc, last_update= 0.0 , 0
    epoch, batch_seen = 0, 0
    auc, val_loss, norm = 0, np.finfo(np.float).max, -1.0
    continue_train = True
    while continue_train:
        epoch += 1
        scheduler.step(auc)
        avg_loss, avg_accuracy = 0.0, 0.0
        for i, batch in enumerate(train_data):
            optimizer.zero_grad()
            batch_loss  = compute_loss(batch, backward = True)
            optimizer.step()

            print(batch_loss)
            avg_loss +=  batch_loss
            update_metrics(batch_loss, 0, key = 'train')

            batch_seen += 1
            if batch_seen % x_batches == 0:
                val_loss, auc = eval_model(val_data)
                update_metrics(val_loss, auc, key = 'val')
                log_metrics()
                logging.info("@Validation round:{}, val_acc:{:.5} val_loss:{:.5}".format(batch_seen/x_batches, val_acc, val_loss))

                norm = 0
                for param in model.parameters():
                    if param.requires_grad:
                        norm += param.norm(2)
                plot_norm(torch.sqrt(norm))

                # save model
                if auc <= best_auc:
                    last_update += 1
                else:
                    best_auc = auc
                    last_update = 0

                    torch.save(model.state_dict(), os.path.join(MODELDIR, "model.torch"))
                    model_state_tmp = dict(config=config, optimizer=optimizer.state_dict(), auc=auc, val_loss=val_loss, finished= not continue_train,\
                                      train_acc=avg_accuracy/(batch_seen+1), train_loss= avg_loss/(batch_seen+1), epoch=epoch, batch_seen=batch_seen)

                    model_state = {}
                    if os.path.isfile(os.path.join(MODELDIR, "model_training.state")):
                        model_state = torch.load(os.path.join(MODELDIR, "model_training.state"), map_location='cpu')
                    model_state.update(model_state_tmp)
                    torch.save(model_state, os.path.join(MODELDIR, "model_training.state"))

                if last_update > 1.0* params.lastupdate or np.isnan(val_loss) or epoch > epoch_limit:
                    continue_train = False
                    break

    logging.info("@epoch:{}, train loss:{:.2}, train accuracy:{:.2},  val loss:{:.2}, \
        val accuracy:{:.2}, param norm:{:.2}".format(epoch,avg_loss/(batch_seen+1), avg_accuracy/(batch_seen+1), val_loss, val_acc, norm))

    model_state = torch.load(os.path.join(MODELDIR, "model_training.state"), map_location='cpu')
    model_state["curr_epoch"] = epoch
    model_state["curr_train_loss"], model_state["curr_train_acc"], = avg_loss/(batch_seen+1), avg_accuracy/(batch_seen+1)
    model_state["last_update"], model_state["finished"] = last_update, not continue_train
    torch.save(model_state, os.path.join(MODELDIR, "model_training.state"))


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_datadir", help="training directory containing folders of soundfiles grouped in their classes e.g. .../age/", required=True)
    parser.add_argument("-val_datadir", help="validation directory containing folders of soundfiles grouped in their classes e.g. .../age/", required=True)
    parser.add_argument("-cuda", help="to run on cuda", default = False)
    parser.add_argument("-pretrained", help="to reload the preexisting model", default = False)
    parser.add_argument("-modeldir", help="directory to store the model. Should contain params.json", default = "")
    args = parser.parse_args()

    cuda = eval(args.cuda) and torch.cuda.is_available() if args.cuda else False
    pretrained = args.pretrained
    MODELDIR = args.modeldir
    torch.save(dict(), os.path.join(MODELDIR, "model_training.state"))

    json_path = os.path.join(MODELDIR, "params.json")
    assert os.path.isfile(json_path), "No cofiguration found at {}".format(json_path)
    params = Params(json_path)

    set_logger(os.path.join(MODELDIR, "train.log"))

    def get_weight_vector():
        if params.dict.get("weightedloss", False):
            return torch.Tensor([5,1,1,1,5,5,10])
        return None

    timeframe = params.timeframe
    windowfn = params.windowfn
    model_arch = params.modelarch
    lr = params.lr
    batch_size = params.batchsize
    x_batches = params.xbatches
    factor = params.schedulerfactor
    patience = params.schedulerpatience
    weight_decay = params.l2
    epoch_limit = params.epoch
    features=params.dict.get("features", "fft")

    window_fn = signal.tukey(51, 0.5)
    if windowfn == "gaussian":
        window_fn = signal.gaussian(51, std=1)

    # load data
    train_data, val_data = AgeDataHandler(args.train_datadir, batch_size).train_val_split()
    logging.info("Number of training observations: {}".format(len(train_data)))
    # import pdb; pdb.set_trace()
    # val_data = AgeDataHandler(args.val_datadir, batch_size)
    logging.info("Number of validation observations: {}".format(len(val_data)))

    epoch_size = int(1.0*len(train_data)/batch_size) + 1
    env_name = params.job_name
    config = dict(lr= lr, batch_size=batch_size, cuda=cuda, \
                  epoch_size=epoch_size, train_data=len(train_data), val_data=len(val_data))
    config.update(params.dict)

    CLASSES = 2 # 0 normal; 1 abnormal
    FINAL_DIM=256
    if params.modelarch == "resnet18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model.fc = torch.nn.Linear(512, FINAL_DIM, bias = True)
    elif params.modelarch == "resnet34":
        model = models.resnet34(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model.fc = torch.nn.Linear(512, FINAL_DIM, bias = True)
    else:
        raise

    model.droprate=0.7
    if params.dict.get('optim',"adam") == "adam":
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
        patience = 50
    elif params.optim == "adamams":
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay, amsgrad=True)
    elif params.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0,weight_decay=weight_decay, nesterov=False)
        patience = 50
    elif params.optim == "sgdnest":
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9,weight_decay=weight_decay, nesterov=True)
        patience = 50
    elif params.optim == "rmsmom":
        optimizer = optim.RMSprop(model.parameters(), lr = lr, weight_decay=weight_decay, momentum=0.9)
    elif params.optim == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr = lr, weight_decay=weight_decay)

    if pretrained:
        assert os.path.isfile("{}/model.torch".format(MODELDIR)), "model not found"
        model.load_state_dict(torch.load("{}/model.torch".format(MODELDIR)))
        optimizer_state = torch.load("{}/model_training.state".format(MODELDIR))['optimizer']
        optimizer.load_state_dict(optimizer_state)
        logging.info("Loading the pre-existing model at {}/model.torch".format(MODELDIR))
    else:
        if os.path.exists(MODELDIR):
            logging.info("Writing in existing directory: {}".format(MODELDIR))
        else:
            raise

    _C_ = torch.zeros(FINAL_DIM)
    if cuda:
        model = model.cuda()
        _C_ = _C_.cuda()

    # compute C
    eps = 1e-3
    ETA = 1.0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for c,batch in enumerate(train_data):
            observations = []
            for i, (soundfile, category) in enumerate(batch):
                Sxx = parse_soundfile(soundfile, timeframe, window_fn, features)
                observations.append(Sxx)

            observations = torch.stack(observations)
            outputs = model(observations)
            n_samples += outputs.shape[0]
            _C_ += torch.sum(outputs, dim = 0)

    _C_ /= n_samples
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    _C_[(abs(_C_) < eps) & (_C_ < 0)] = -eps
    _C_[(abs(_C_) < eps) & (_C_ > 0)] = eps

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, verbose=True)
    update_metrics, log_metrics, plot_norm = init_visdom(env_name, config)

    model.train()
    train(model, train_data, optimizer)
