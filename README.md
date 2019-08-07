# ResNet on Audio for Age Classification

## How to run a single experiment
`train.py` is a training script for age classification.
`train_anom.py` is a training script which treats ages above 70 as an anomaly.
Both the scripts takes the following arguments -  

```
-train_datadir TRAIN_DATADIR
                       training directory containing folders of soundfiles
                       grouped in their classes e.g. .../age/
 -cuda CUDA            to run on cuda
 -pretrained PRETRAINED
                       to reload the preexisting model
 -modeldir MODELDIR    directory to store the model. Should contain
                       params.json
```

`MODELDIR` is the directory where the `params.json` should be present.
`params.json` defines training configuration as well as model hyperparameters.

Check `run.sh` for an example.

## Data

Data directory is assumed to have the following structure
```
Common_voice
|_ Common_Voice_train
  |_male
    |_ages
      |_ ...
```

`data_cleaning.py` converts the data to `.wav` format and deletes the `.mp3` version.

## Prerequisites
The Python libraries used in this project are in `requirements.txt`
Additionally, the library - `mlogger` used for logging the training progress in `visdom` is from a specific commit. Follow these steps to install the mlogger -

```
git clone https://github.com/oval-group/mlogger.git
cd mlogger
git checkout 4623d3446b5a4223a158b3cf9379ec0c065183a4
python setup.py develop
```

If `mlogger` is in the repo, you can also just  `python setup.py develop` instead of going through the above steps.

## Hyperparameter Tuning
The only thing needed to run hyperparameter tuning is to write a python script like `run-gausswindow.py` and then run the script generated from it.

`run-gausswindow.py` is a script to generate `run_exp-GAUSSIANres18.sh`. The python script calls `generate_shell_script()` in `shell_utils.py` to generate a list of python commands needed to run for the experiment.

Use the command below to tabulate results from the above experiments. Example output is in `results-GAUSSIANres18`
```
python shell_utils.py --show_results -id=GAUSSIANres18
```

## Troubleshooting

* If `scipy.signal` aborts, try `export KMP_DUPLICATE_LIB_OK=TRUE` before running the script.
