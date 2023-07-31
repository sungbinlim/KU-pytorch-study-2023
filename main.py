import torch
import torch.nn as nn
import torch.optim as optim
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from util import *
from trainer import train_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hyperparameter
data_dir = './ion_data'
batch_size = 512
train_data = IonDataset(data_dir, 'train')
valid_data = IonDataset(data_dir, 'valid')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

lr = 0.1
epochs = 30
model = IonPredictor().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_classifier = nn.CrossEntropyLoss()
loss_regression = nn.MSELoss()

train_step = make_train_step(model, train_loss_fn, optimizer)
valid_step = make_valid_step(model, valid_loss_fn)
train_loss, valid_loss = train_model(train_dataloader, valid_dataloader, epochs=epochs, checkpoint=True, device=device)