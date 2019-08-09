# Use of Audio Features from Age Detector to Diagnose Cognitive Impairment

## Data Quality in Healthcare
Data collection in healthcare is expensive and time consuming.
Most importantly, it should be done with utmost care because a poor data quality will only push back the progress of AI in healthcare.
The machine learning community has been working hard to beat the benchmarks on the healthcare datasets.
However, a recent research [1] revisited some of these healthcare datasets to evaluate their quality which wasn't found to be at par with experts.

On the other hand, collecting data in healthcare for rare diseases just takes time.
For example, if one needs to study the correlation between the changes in voice patterns and the decline of cognitive status in an Alzheimer's patient, the study needs to be done across several years to uncover any meaningful insights.
And this needs to be done for several patients.
And it needs to be done with high quality microphones so that the models can be built upon them.
All these factors add to the cost of data collection in healthcare, and is not always possible.
And there are many changes that occur across the time span of this process, like installing good microphones halfway into the study.
Thus, we need ways to deal with such scarcity as well as the noise in datasets.  

## What this repo is about?
This repo is a collaboration on a project to build an end-to-end deep learning classifier to detect mild cognitive impairment from a dataset which has a very few observations compared to what are required in such highly parameterized networks.
The dataset has interview recordings of less than 150 patients taken periodically across 30 years.
Some of them are controls, and some of them are patients who are at the some stage of Alzheimer's.
At every visit, the patient is given some task to test their cognitive ability, and finally, a cognitive status score is given by an expert at the end of this visit.

A team of researchers have done a good job in building machine learning models with manual features to ace the dataset. The model has an AUC of 0.92.
However, one needs to take into account several other factors while building such a learning system.
- All of the patients in the study belonged to the same geographical area. Therefore, the models are less likely to generalize to patients from different demography.
- Manual feature engineering is not always feasible as there are chances of leaving out some crucial features or interactions thereof.

This project aims at applying ** transfer learning to make an end-to-end deep learning classifier for Alzheimer's**.
Unfortunately, the data is not open sourced, so one can't run some of these experiments at their end, but we aim to display results and discuss some shortcomings that we faced.

## Age Classifier
We use Resnet [2] architecture on audio data from [CommonVoice](https://voice.mozilla.org/en) to classify age.
We found the best performance (about 85% weighted accuracy) of the model when the audio samples were transformed into their MFCC format.
Further details on training method can be found in `train.py`.
Our aim is to extract features from the trained model to further use them in the classifier to be built for the Alzheimer's dataset.
However, a good performing model is not of great use because of several reasons as mentioned below.

**Pitfall 1:** The CommonVoice dataset is highly skewed towards the ages between 20-40. There are a few samples of above 70s which is what we would finally like to apply on.

**Pitfall 2:** The interview recordings are not an easy dataset to handle.  It is interweaved with voices of an interviewee and the interviewer. This calls for an accurate diarization techniques.

**Pitfall 3:** The interview recordings are across a span of 30 years. There are a lot of changes in the quality of microphones and the settings. Some recordings have a lot of noise from the fan behind.

**Pitfall 4:** Not all interview days are same. It might be that some days the patients are just not willing to speak much. Thus, one can't rely on content, but only the voice markers or prosodic features.

As a workaround, we tried treating older age to be an anomaly by using DeepSAD [3] in the hope of learning more useful features.
The model that treats older age as anomaly got an AUC of 0.8 for male and 0.6 for female.
As evident from the AUC, we don't expect to have good features here.
Further details on this training model can be found in `train_anom.py`.

## Transfer Learning
In the second stage of the project, we extracted the features from the above models to check the performance on the smaller Alzheimer's dataset.
Due to improper diarization of the dataset, we had to focus on just the Story Recall Test where we could obtain a continuous 20 second audio sample.
This project treats all the observations as independent (see Caveat 1 below).
As the training results weren't conclusive, we don't report any numbers here.
The code for transfer learning is in `train_transfer.py`.
The script implements two loss functions - (i) ranking loss similar to SVMs, (ii) cross-entropy loss.
`run_transfer.sh` provides an example on how to run this script.

**Caveat 1:** Since the dataset contains a longitudinal study of patients being interviewed one needs to be careful about how to split the dataset. The observations in the dataset are not independent as there can be multiple recrodings of the same person. The 2017 survey [4] on data splitting techniques will be of help.

**Caveat 2:** This project used only the last layer as features, but it is possible to use other layers as features as well.

## How to run a single experiment
`train.py` is a training script for age classification.
`train_anom.py` is a training script which treats ages above 70 as an anomaly.
They take the following arguments -  

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
A train and test split on train_datadir is performed in the script itself.
A seed is set in the code to result in the same split every time.
Check `run.sh` for an example to run the code.
`confusion.py` computes confusion matrix and outputs several performance metrics.

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

## References
[1] Oakden-Rayner, L. (2019). Exploring large scale public medical image datasets, 14. Retrieved from http://arxiv.org/abs/1907.12720

[2] Hembury, G. A., Borovkov, V. V., Lintuluoto, J. M., & Inoue, Y. (2003). Deep Residual Learning for Image Recognition Kaiming. CVPR, 32(5), 428–429. https://doi.org/10.1246/cl.2003.428

[3] Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E., Müller, K.-R., & Kloft, M. (2019). Deep Semi-Supervised Anomaly Detection, http://arxiv.org/abs/1906.02694

[4] Little, M. A., Varoquaux, G., Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). Using and understanding cross-validation strategies. Perspectives on Saeb et al. GigaScience, 6(5), 1–6. doi:10.1093/gigascience/gix020
