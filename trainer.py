import torch
import torch.nn as nn
import torch.optim as optim
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_fn(predict, y):

    ion_number_target, potential_target = y[0], y[1]
    ion_number_predict, potential_predict = predict[0], predict[1]        
    loss = loss_classifier(ion_number_predict, ion_number_target) + loss_regression(potential_predict, potential_target)
    return loss

def train_model(train_loader, valid_loader, epochs=100):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for x_train_batch, y_train_batch_ion, y_train_batch_potential in train_loader:
            x_train_batch = x_train_batch.to(device)
            y_train_batch_ion = y_train_batch_ion.to(device)
            y_train_batch_potential = y_train_batch_potential.to(device)
            loss = train_step(x_train_batch, [y_train_batch_ion, y_train_batch_potential])
            train_loss += loss
        train_losses.append(train_loss / len(train_data))
            
        # Evaluate train loss
        if epoch % 10 == 0:
            print("train loss at {} epoch:{}".format(epoch, train_loss))

        # Evaluate valid loss
            with torch.no_grad():
                valid_loss = 0
                for x_valid_batch, y_valid_batch_ion, y_valid_batch_potential in valid_loader:
                    x_valid_batch = x_valid_batch.to(device)
                    y_valid_batch_ion = y_valid_batch_ion.to(device)
                    y_valid_batch_potential = y_valid_batch_potential.to(device)
                    eval_valid_loss = valid_step(x_valid_batch, [y_valid_batch_ion, y_valid_batch_potential])
                    valid_loss += eval_valid_loss
                valid_loss = valid_loss / len(valid_data)
                print("valid loss at {} epoch:{}".format(epoch, valid_loss))
                valid_losses.append(valid_loss)

    return valid_losses

# Hyperparameter
data_dir = './ion_data'
batch_size = 256
lr = 0.1
epochs = 100
model = IonPredictor().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_classifier = nn.CrossEntropyLoss()
loss_regression = nn.MSELoss()

train_step = make_train_step(model, loss_fn, optimizer)
valid_step = make_valid_step(model, loss_fn)

train_data = IonDataset(data_dir, 'train')
valid_data = IonDataset(data_dir, 'valid')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

valid_loss = train_model(train_dataloader, valid_dataloader)
# print(valid_loss)