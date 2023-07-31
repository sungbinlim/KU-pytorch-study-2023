import torch
import torch.nn as nn
import torch.optim as optim
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from util import *

def train_model(train_loader, valid_loader, epochs=100, checkpoint=False, device='cpu'):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_loss = 0

        for x_train_batch, y_train_batch_ion, y_train_batch_potential in train_loader:
            # 데이터 로더에서 받은 미니배치를 device 에 업로드
            x_train_batch = x_train_batch.to(device)
            y_train_batch_ion = y_train_batch_ion.to(device)
            y_train_batch_potential = y_train_batch_potential.to(device)

            # 미니매치 데이터를 이용해 parameter update
            loss = train_step(x_train_batch, [y_train_batch_ion, y_train_batch_potential])
            train_loss += loss
        train_loss = train_loss / len(train_data)    
        train_losses.append(train_loss)
            
        # Evaluate train loss
        if epoch % 5 == 0:
            print("train loss at {} epoch:{}".format(epoch, train_loss))

        # Evaluate valid loss
            with torch.no_grad():
                valid_loss = 0
                cnt = 0
                for x_valid_batch, y_valid_batch_ion, y_valid_batch_potential in valid_loader:
                    # 데이터 로더에서 받은 미니배치를 device 에 업로드
                    x_valid_batch = x_valid_batch.to(device)
                    y_valid_batch_ion = y_valid_batch_ion.to(device)
                    y_valid_batch_potential = y_valid_batch_potential.to(device)
                    
                    # 미니매치 데이터를 이용해 performance 평가
                    _, eval_valid_loss_regressor, correct_cnt = valid_step(x_valid_batch, [y_valid_batch_ion, y_valid_batch_potential])
                    valid_loss += eval_valid_loss_regressor
                    cnt += correct_cnt
                valid_loss = valid_loss / len(valid_data)
                cnt = 100 * cnt / len(valid_data)
                print("valid MSE loss and accuracy at {} epoch:{} and {}%".format(epoch, valid_loss, cnt))
                valid_losses.append(valid_loss)
            
            if checkpoint:
                checkpoint = {'epochs': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'test_loss': valid_losses
                }

                torch.save(checkpoint, 'model_checkpoint.pth')

    return train_losses, valid_losses

