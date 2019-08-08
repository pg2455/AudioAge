import pandas as pd
from base_utils import parse_soundfile
from scipy import signal
from torchvision import models
import torch, os
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from train_utils import AgeDataHandler

np.random.seed(1)
torch.manual_seed(1)

DATA = "../Common_voice/Common_Voice_train/male/ages/"
MODELDIR = "../exp1"

# load model
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
model.fc = nn.Linear(512, 7, bias = True)
model.load_state_dict(torch.load("{}/model.torch".format(MODELDIR), map_location="cpu"))

_, val_data = AgeDataHandler(DATA).train_val_split()
y_true, y_pred = [], []
for batch in val_data:
    observations = []
    for i, (soundfile, category) in enumerate(batch):
        Sxx = parse_soundfile(soundfile, timeframe=20, features="mfcc")
        observations.append(Sxx)
        y_true.append(category)
    output = model(torch.stack(observations))
    pred = torch.argmax(output, dim=1)
    y_pred += list(pred)
print(confusion_matrix(y_true, y_pred))
print("micro", precision_recall_fscore_support(y_true, y_pred, average='micro'))
print("macro", precision_recall_fscore_support(y_true, y_pred, average='macro'))
print("weighted", precision_recall_fscore_support(y_true, y_pred, average='weighted'))
