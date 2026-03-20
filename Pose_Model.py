import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTConfig
from torchvision.models import resnet18, ResNet18_Weights


class CNN3(nn.Module):
    def __init__(self, pitch_num_classes, roll_num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512), nn.ReLU()
        )  # Assuming input size of 224x224
        self.pitch_classifier = nn.Linear(512, pitch_num_classes)
        self.roll_classifier = nn.Linear(512, roll_num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flat(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        pitch_logits = self.pitch_classifier(x)
        roll_logits = self.roll_classifier(x)
        return pitch_logits, roll_logits
        


class VGG(nn.Module):
    def __init__(self, pitch_num_classes, roll_num_classes):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        num_ftrs = self.vgg.classifier[-1].in_features
        self.vgg.classifier = nn.Identity()
        self.pitch_classifier = nn.Linear(num_ftrs, pitch_num_classes)
        self.roll_classifier = nn.Linear(num_ftrs, roll_num_classes)

    def forward(self, x):
        x = self.vgg(x)
        pitch_logits = self.pitch_classifier(x)
        roll_logits = self.roll_classifier(x)
        return pitch_logits, roll_logits


class Resnet18(nn.Module):
    def __init__(self, pitch_num_classes, roll_num_classes):
        super(Resnet18, self).__init__()
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.pitch_classifier = nn.Linear(num_ftrs, pitch_num_classes)
        self.roll_classifier = nn.Linear(num_ftrs, roll_num_classes)

    def forward(self, x):
        x = self.resnet(x)
        pitch_logits = self.pitch_classifier(x)
        roll_logits = self.roll_classifier(x)
        return pitch_logits, roll_logits

class Resnet50(nn.Module):
    def __init__(self, pitch_num_classes, roll_num_classes):
        super(Resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.pitch_classifier = nn.Linear(num_ftrs, pitch_num_classes)
        self.roll_classifier = nn.Linear(num_ftrs, roll_num_classes)

    def forward(self, x):
        x = self.resnet(x)
        pitch_logits = self.pitch_classifier(x)
        roll_logits = self.roll_classifier(x)
        return pitch_logits, roll_logits

class VisionTransformer(nn.Module):
    def __init__(self, pitch_num_classes, roll_num_classes):
        super(VisionTransformer, self).__init__()
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        num_ftrs = self.vit.config.hidden_size
        self.pitch_classifier = nn.Linear(num_ftrs, pitch_num_classes)
        self.roll_classifier = nn.Linear(num_ftrs, roll_num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # CLS token output
        pitch_logits = self.pitch_classifier(features)
        roll_logits = self.roll_classifier(features)
        return pitch_logits, roll_logits

