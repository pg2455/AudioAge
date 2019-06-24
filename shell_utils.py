import json, os, torch
from configuration import train_dir, val_dir, base_modeldir, CUDA
from base_utils import Params
import pandas as pd
lr_values = [1e-4, 1e-3]
l2_values = [1e-3, 1e-2, 1e-1, 1e0]

def get_lr_l2_combinations(lr_values = lr_values, l2_values = l2_values):
    hyperparams = []
    for lr in lr_values:
        for l2 in l2_values:
            hyperparams.append( [('lr', lr), ('l2', l2)] )
    return hyperparams


class FixedDict(object):
        def __init__(self, dictionary):
            self._dictionary = dictionary
        def __setitem__(self, key, item):
                if key not in self._dictionary:
                    raise KeyError("The key {} is not defined.".format(key))
                self._dictionary[key] = item
        def __getitem__(self, key):
            return self._dictionary[key]

        def dict(self):
            return self._dictionary

def generate_shell_script(base_params, id, lr_values=lr_values, l2_values= l2_values, CUDA=CUDA):
    script = "export KMP_DUPLICATE_LIB_OK=TRUE\n"
    parent_dir = os.path.join(base_modeldir, "experiments", "experiment_{}".format(id))
    os.makedirs(parent_dir)
    base_json_path = os.path.join(parent_dir, "base_params.json")
    json.dump(base_params.dict(), open(base_json_path, "w"))

    # assumption - 1: parameter names doesnt have any - or _. They are just one word fields.
    # assumption - 2: The values are of the format 1e..
    hyperparams = get_lr_l2_combinations(lr_values, l2_values)
    for hyperparam in hyperparams:
        params = Params(base_json_path)
        job_name = []
        for (n,v) in hyperparam:
            job_name += ["{}-{:.0e}".format(n,v)]
            params.dict[n] = v
        job_name = "_".join(job_name)
        modeldir = os.path.join(parent_dir, *job_name.split("_"))
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        params.job_name = "{}-{}".format(id, job_name)
        json_path = os.path.join(modeldir, 'params.json')
        params.save(json_path)


        script += "python train.py -train_datadir={} -val_datadir={} -cuda={} -modeldir={} \nwait\n".format(train_dir, val_dir, CUDA, modeldir)

    with open("run_exp-{}.sh".format(id), 'w') as f:
        f.write(script)

def tabulate_results(id):
    experiment_dir = os.path.join(base_modeldir, "experiments", "experiment_{}".format(id))
    hyperparams = [('','lr'), ('', 'l2')]
    columns = hyperparams + [('val', 'l'),('val', 'A'), ('train', 'l'),('train', 'A'), ('','epoch')] + \
                        [('curr_train', 'l'), ('curr_train', 'A')] + [('','curr_epoch'), ('','last_update'), ('','batch_seen'), ('','finished')]

    base_params = Params(os.path.join(experiment_dir, 'base_params.json'))
    table = []
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            if file == "model_training.state":
                path = os.path.join(root, file)
                state = torch.load(path, map_location='cpu')
                val_loss, val_acc = state['val_loss'], state['val_acc']
                train_loss, train_acc = state['train_loss'], state['train_acc']
                epoch, batch_seen = state['epoch'], state.get('batch_seen', -1)

                #
                curr_train_loss, curr_train_acc = state.get('curr_train_loss',None), state.get('curr_train_acc',None)
                curr_epoch, last_update, finished = state.get('curr_epoch', None), state.get('last_update',None), state.get('finished', None)

                params = Params(os.path.join(root, "params.json"))
                row = [params.dict[x.split("-")[0]] for x in root.replace(experiment_dir, "").split('/') if x]
                row += list(map(lambda x: round(x,2), [val_loss, val_acc, train_loss, train_acc]))  + [epoch]
                row += list(map(lambda x: round(x,2) if x else None, [curr_train_loss, curr_train_acc])) + [curr_epoch, last_update, batch_seen, finished]
                table.append(row)

    df = pd.DataFrame(table, columns=columns)
    df.sort_values(hyperparams, inplace=True)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index = range(len(df))
    # caption = ", ".join(["{}:{}".format(k,v) for k,v in base_params.dict.iteritems() if k!='job_name'])
    # tex = df.to_latex().replace("\n", "\n\\caption{{{}}}\\\\\n".format(caption),1)
    # csv = df.to_csv()
    with open("results-{}".format(id),'w') as f:
        f.write(df.to_string())
    print(df.to_string())
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--show_results",  action="store_true", help="show results for the experiment in id' ", dest='show_results')
    parser.add_argument('-id', default='', help="experiment id")
    args = parser.parse_args()

    if args.show_results:
        tabulate_results(args.id)
