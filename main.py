import os
import pandas as pd
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.dataset import CropDiseaseDataset, DatasetInitializer
from torchsummary import summary
from src.models import SqueezeNetObjectDetectionWithFPN,  SqueezeNetObjectDetectionWithConfidence
from src.losses import ObjectDetectionLoss
from torch.utils.data import DataLoader
from src.trainer import Trainer
from src.utils import plot_losses
from src.evaluator import Evaluator

wandb.login(key='5e145e48a5fee4de9ed324a009bd7e5b2ced1eee')
wandb.init(project="crop-disease-detection", entity="beriakalpelbe")

# Configurations
config = wandb.config
config.batch_size = 128
config.learning_rate = 1e-3
config.num_epochs = 50
config.num_classes = 23
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.num_boxes = 50

# Model Initialization
dataset_initializer = DatasetInitializer()
dataset_train = CropDiseaseDataset(dataframe=dataset_initializer.X_train, num_boxes=config.num_boxes)
dataset_val = CropDiseaseDataset(dataframe=dataset_initializer.X_val, num_boxes=config.num_boxes)
dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
# model = SqueezeNetObjectDetectionWithConfidence(num_classes=num_classes, num_boxes=num_boxes)
model = SqueezeNetObjectDetectionWithFPN(num_classes=config.num_classes, num_boxes=config.num_boxes)
model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = ObjectDetectionLoss(num_classes=config.num_classes, num_boxes=config.num_boxes)
print(summary(model, input_size=(3, 224, 224)))

# Model training
trainer = Trainer(model, optimizer, criterion, config)
trainer.run(dataloader_train, dataloader_val)
plot_losses(trainer.train_losses, trainer.val_losses)

# Model evaluation
evaluator = Evaluator(model, config, dataset_initializer.X_val)