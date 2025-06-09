import os
import random
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from tqdm import tqdm
from fimage import FImage
from fimage.filters import (
    Brightness, Contrast, Exposure, Grayscale, Hue,
    Noise, Posterize, Saturation, Sepia, Sharpen, Vibrance
)

FILTERS = ['Contrast', 'Brightness', 'Saturation', 'Hue', 'Exposure', 'Vibrance', 'Sepia', 'Sharpen', 'Noise']

class MultiOutputEfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MultiOutputEfficientNet, self).__init__()
        self.base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        in_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Identity()

        self.filter_heads = nn.ModuleDict()
        for i in range(len(FILTERS)):
            self.filter_heads[FILTERS[i]] = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        features = self.base_model(x)
        
        outputs = []
        for head in self.filter_heads.values():
            outputs.append(head(features))
        
        return torch.cat(outputs, dim=1)