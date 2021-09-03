"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image

import os

import torch
import torchvision
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

lb = pickle.load(open('/content/drive/MyDrive/DEEPVIS/label_encoder.pkl', 'rb'))
lb_OD = pickle.load(open('/content/drive/MyDrive/DEEPVIS/label_encoder_OD.pkl', 'rb'))


def get_faster_rcnn():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(lb_OD.classes_) + 1 
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class RegNet(nn.Module):
  def __init__(self, input_shape=(1, 256, 256)):
    super().__init__()
    # Here, we define all the weights for the neural network, they are abstracted by layers. Internally however, they are represented by Pytorch tensors.
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.batchnorm3 = nn.BatchNorm2d(128)

    self.pool = nn.MaxPool2d(kernel_size=2)

    n_size = self._get_conv_output(input_shape)

    self.fc1 = nn.Linear(in_features=n_size, out_features=512)
    self.fc2 = nn.Linear(in_features=512, out_features=len(lb.classes_))

    self.dropout = nn.Dropout(0.4)

  def _get_conv_output(self, shape):
    batch_size = 1
    input = torch.autograd.Variable(torch.rand(batch_size, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size
  
  def _forward_features(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    return x
  
  def forward(self, x):
      # Here, we define how an input x is translated into an output. In our linear example, this was simply (x^T * w), now it becomes more complex but
      # we don't have to care about that (gradients etc. are taken care of by Pytorch).
      x = self._forward_features(x)
      # You can always print shapes and tensors here. This is very very helpful to debug.
      # print("x.shape:", x.shape)
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      return x

