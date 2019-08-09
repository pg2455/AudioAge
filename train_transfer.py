import pandas as pd
from scipy import signal
from torchvision import models
import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import logging
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from train_utils import FileIterator, init_visdom, AnomModel
from base_utils import Params, parse_soundfile, set_logger

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

def compute_loss(batch, backward = False, anomaly=False):
    observations = []
    target = torch.zeros(len(batch), 2)
    for i, (soundfile, score) in enumerate(batch):
        Sxx = parse_soundfile(soundfile, timeframe, window_fn, features)
        observations.append(Sxx)
        category = 1 if score > 2 else 0
        target[i,category] = 1.0

    observations = torch.stack(observations)
    if cuda:
        observations = observations.cuda()
        target = target.cuda()

    output = model(observations)
    loss = loss_fn(output, target)
    if backward:
        loss.backward()

    loss = loss.detach().item() / len(batch)
    acc = 1.0 * (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).sum().item() / len(batch)

    return loss, acc, torch.argmax(target.detach(), dim=1).cpu().numpy().tolist(), output.detach()[:,1].cpu().numpy().tolist()

def eval_model(val_data):
    total_loss, total_acc, total_obs = 0, 0, 0
    targets, scores = [], []
    for i, batch in enumerate(val_data):
        batch_loss, batch_acc, y, s  = compute_loss(batch, backward = True)
        total_loss +=  batch_loss * len(batch)
        total_acc += batch_acc *  len(batch)
        total_obs += len(batch)
        targets += y
        scores += s

    auc = roc_auc_score(targets, scores)
    return total_loss/total_obs, total_acc/total_obs, auc

def train(model, train_data, optimizer):
    best_acc, last_update= 0.0 , 0
    epoch, batch_seen = 0, 0
    val_acc, val_loss, norm = -1.0, np.finfo(np.float).max, -1.0
    continue_train = True
    while continue_train:
        epoch += 1
        scheduler.step(val_acc)
        avg_loss, avg_accuracy = 0.0, 0.0
        for i, batch in enumerate(train_data):

            optimizer.zero_grad()
            batch_loss, batch_acc,_,_  = compute_loss(batch, backward = True)
            optimizer.step()

            print(batch_loss, batch_acc)
            avg_loss +=  batch_loss
            avg_accuracy += batch_acc
            update_metrics(batch_loss, batch_acc, key = 'train')

            batch_seen += 1
            if batch_seen % x_batches == 0 and len(val_data) > 0:
                val_loss, val_acc, auc = eval_model(val_data)
                update_metrics(val_loss, auc, key = 'val')
                log_metrics()
                logging.info("@Validation round:{}, val_acc:{:.5} val_loss:{:.5} val_auc:{:.5}".format(batch_seen/x_batches, val_acc, val_loss, auc))

                norm = 0
                for param in model.parameters():
                    if param.requires_grad:
                        norm += param.norm(2)
                plot_norm(torch.sqrt(norm))

                # save model
                if val_acc <= best_acc:
                    last_update += 1
                else:
                    best_acc = val_acc
                    last_update = 0

                    torch.save(model.state_dict(), os.path.join(MODELDIR, "model.torch"))
                    model_state_tmp = dict(config=config, optimizer=optimizer.state_dict(), val_acc=val_acc, val_loss=val_loss, finished= not continue_train,\
                                      train_acc=avg_accuracy/(batch_seen+1), train_loss= avg_loss/(batch_seen+1), epoch=epoch, batch_seen=batch_seen)

                    model_state = {}
                    if os.path.isfile(os.path.join(MODELDIR, "model_training.state")):
                        model_state = torch.load(os.path.join(MODELDIR, "model_training.state"), map_location='cpu')
                    model_state.update(model_state_tmp)
                    torch.save(model_state, os.path.join(MODELDIR, "model_training.state"))

                if last_update > 1.0* transfer_model_params.lastupdate or np.isnan(val_loss) or epoch > epoch_limit:
                    continue_train = False
                    break

    logging.info("@epoch:{}, train loss:{:.2}, train accuracy:{:.2},  val loss:{:.2}, \
        val accuracy:{:.2}, param norm:{:.2}".format(epoch,avg_loss/(batch_seen+1), avg_accuracy/(batch_seen+1), val_loss, val_acc, norm))

    model_state = torch.load(os.path.join(MODELDIR, "model_training.state"), map_location='cpu')
    model_state["curr_epoch"] = epoch
    model_state["curr_train_loss"], model_state["curr_train_acc"], = avg_loss/(batch_seen+1), avg_accuracy/(batch_seen+1)
    model_state["last_update"], model_state["finished"] = last_update, not continue_train
    torch.save(model_state, os.path.join(MODELDIR, "model_training.state"))


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda", help="to run on cuda", default = False)
    parser.add_argument("-gender", help="gender", default = "male")
    parser.add_argument("-base_modeldir", help="directory of the base model. Should contain model.torch and params.json", default = "")
    parser.add_argument("-modeldir", help="directory of the new transferred model. Should contain params.json", default = "")
    parser.add_argument("-anomaly", help="if the base model is anomaly detector", default = False)
    args = parser.parse_args()

    cuda = eval(args.cuda) and torch.cuda.is_available() if args.cuda else False
    BASE_MODELDIR = args.base_modeldir
    MODELDIR = args.modeldir
    set_logger(os.path.join(MODELDIR, "train.log"))
    GENDER = args.gender

    json_path = os.path.join(BASE_MODELDIR, "params.json")
    assert os.path.isfile(json_path), "No cofiguration found at {}".format(json_path)
    base_params = Params(json_path)

    json_path = os.path.join(MODELDIR, "params.json")
    assert os.path.isfile(json_path), "No cofiguration found at {}".format(json_path)
    transfer_model_params = Params(json_path)

    timeframe = base_params.timeframe
    windowfn = base_params.windowfn
    model_arch = base_params.modelarch
    window_fn = signal.tukey(51, 0.5)
    if windowfn == "gaussian":
        window_fn = signal.gaussian(51, std=1)

    batch_size = transfer_model_params.batchsize
    x_batches = transfer_model_params.xbatches
    factor = transfer_model_params.schedulerfactor
    patience = transfer_model_params.schedulerpatience
    features = transfer_model_params.dict.get("features", "fft")
    lr = transfer_model_params.lr
    weight_decay = transfer_model_params.l2
    epoch_limit = transfer_model_params.epoch

    MASTER_TABLE = "../master_table.csv"
    FHS_DATA = "../Patient Story Test Voice Samples"

    def loss_fn(output, target):
        # ranking loss; separate out controls from target i.e. learn a representation that separtes them most
        P = (target[:,1] == 1).nonzero().flatten()
        N = (target[:,1] !=1).nonzero().flatten()
        score_n = output[N,1].repeat(len(P))
        score_p = output[P,1].unsqueeze(1).repeat(1, len(N)).view(-1,1).squeeze()
        loss = torch.clamp(1 + score_n - score_p, min=0).sum()
        return loss/(P.numel() * N.numel())
        # cross entropy loss
        # criterion = torch.nn.BCEWithLogitsLoss()
        # return criterion(output, target)

    # load data
    data = pd.read_csv("../master_table.csv")
    story_data = data[data['test'] == "story_test"][['mostSevereCogStatus_atTimeOfExam', 'ranid',  'NP Exam Date', 'Age', 'SEX']]

    data = []
    for story_file in os.listdir(FHS_DATA):
        ranid = int(story_file.split("_")[1])
        date = story_file.split("_")[2]
        mm, dd, yy = date[:2], date[2:4], date[4:]
        filedate = "20{}-{}-{}".format(yy, mm, dd)
        soundfile = os.path.join(FHS_DATA, story_file)
        is_present = (story_data['ranid'] == ranid) * (story_data['NP Exam Date'] ==  filedate)
        gender = story_data[is_present]["SEX"].unique()
        if is_present.sum() > 0 and gender and gender.item() == GENDER:
            score = story_data[is_present]["mostSevereCogStatus_atTimeOfExam"].unique().item()
            if score >= 2 or score < 2:
                data.append((soundfile, score))

    # prepare data
    random.shuffle(data)
    train_data, val_data = train_test_split(data, test_size=0.3)
    train_data, val_data = FileIterator(train_data, 64), FileIterator(val_data, 64)

    controls = sum([1 for x in train_data.tolist() if x[1] < 2])
    subjects = sum([1 for x in train_data.tolist() if x[1] >= 2])
    print("Train - Number of subjects: {}, Number of control:{}".format(subjects, controls))

    controls = sum([1 for x in val_data.tolist() if x[1] < 2])
    subjects = sum([1 for x in val_data.tolist() if x[1] >= 2])
    print("Val - Number of subjects: {}, Number of control:{}".format(subjects, controls))


    epoch_size = int(1.0*len(train_data)/batch_size) + 1
    env_name = transfer_model_params.job_name
    config = dict(lr= lr, batch_size=batch_size, cuda=cuda, \
                  epoch_size=epoch_size, train_data=len(train_data), val_data=len(val_data))
    config.update(transfer_model_params.dict)

    # load model
    kernel_size=7 if eval(args.anomaly) else 8
    FINAL_DIM = 256 if eval(args.anomaly) else 8
    if model_arch == "resnet18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2, padding=3,bias=False)
        model.fc = torch.nn.Linear(512, FINAL_DIM, bias = True)
    elif model_arch == "resnet34":
        model = models.resnet34(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2, padding=3,bias=False)
        model.fc = torch.nn.Linear(512, FINAL_DIM, bias = True)
    else:
        raise

    model.load_state_dict(torch.load("{}/model.torch".format(BASE_MODELDIR), map_location="cpu"))

    # fix layers
    for param in model.parameters():
        param.requires_grad = False

    if eval(args.anomaly):
        model = AnomModel(model)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, 2))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if cuda:
        model = model.cuda()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20, verbose=True)
    update_metrics, log_metrics, plot_norm = init_visdom(env_name, config)

    train(model, train_data, optimizer)
