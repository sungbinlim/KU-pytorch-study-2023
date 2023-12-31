import torch
import torch.nn as nn
import torch.optim as optim
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from util import *
import time
import wandb

# Config file
config = {
        "lr": 0.1,
        "epochs": 40,
        "batch_size": 512,
        "data_dir": './ion_data',
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "model": IonPredictor,
        "optimizer": optim.SGD,
        "loss_classifier": nn.CrossEntropyLoss,
        "loss_regression": nn.MSELoss,
        }

loss_classifier = config['loss_classifier']()
loss_regression = config['loss_regression']()

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {computation_time} seconds")
        return result
    return wrapper

def valid_loss_fn(predict, y):

    ion_number_target, potential_target = y[0], y[1]
    ion_number_predict, potential_predict = predict[0], predict[1]        
    return loss_classifier(ion_number_predict, ion_number_target), loss_regression(potential_predict, potential_target)

def train_loss_fn(predict, y):

    ion_number_target, potential_target = y[0], y[1]
    ion_number_predict, potential_predict = predict[0], predict[1]        
    loss = loss_classifier(ion_number_predict, ion_number_target) + loss_regression(potential_predict, potential_target)
    return loss

def train_model(train_loader, valid_loader, epochs=100, checkpoint=False, device='cpu'):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):

        for x_train_batch, y_train_batch_ion, y_train_batch_potential in train_loader:
            # 데이터 로더에서 받은 미니배치를 device 에 업로드
            x_train_batch = x_train_batch.to(device)
            y_train_batch_ion = y_train_batch_ion.to(device)
            y_train_batch_potential = y_train_batch_potential.to(device)

            # 미니매치 데이터를 이용해 parameter update
            loss = train_step(x_train_batch, [y_train_batch_ion, y_train_batch_potential])
            wandb.log({'train_loss': loss}, step=epoch)  
            
        train_losses.append(loss)
        
        # Evaluate train loss
        if epoch % 10 == 0:
            print("Epoch:{} / Train loss: {}".format(epoch, loss))

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
                    valid_loss += eval_valid_loss_regressor * len(x_valid_batch)
                    cnt += correct_cnt
                valid_loss = valid_loss / len(valid_data)
                valid_losses.append(valid_loss)
                accuracy = 100 * cnt / len(valid_data)
                
                print("Epoch: {} / Valid MSE loss: {} / Accuracy {} %".format(epoch, valid_loss, accuracy))
                wandb.log({'valid_MSE_loss': valid_loss}, step=epoch)  
                wandb.log({'valid_accuracy': accuracy}, step=epoch)
            
            if checkpoint:
                checkpoint = {'epochs': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'valid_loss': valid_losses
                }

                torch.save(checkpoint, 'model_checkpoint.pth')

    return train_losses, valid_losses

if __name__ == "__main__":
    
    # Hyperparameters
    data_dir = config['data_dir']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    device = config['device']
    model = config['model']().to(device)
    optimizer = config['optimizer'](model.parameters(), lr=lr)
    loss_classifier = config['loss_classifier']()
    loss_regression = config['loss_regression']()

    # Data setting
    train_data = IonDataset(data_dir, 'train')
    valid_data = IonDataset(data_dir, 'valid')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    wandb.login()
    wandb.init(project="wandb-ion-project-test",  
               name=f"experiment_{config['lr']}_{config['epochs']}_{config['batch_size']}", 
               config=config,  
               )
    
    train_step = make_train_step(model, train_loss_fn, optimizer)
    valid_step = make_valid_step(model, valid_loss_fn)
    train_loss, valid_loss = train_model(train_dataloader, valid_dataloader, epochs=epochs, checkpoint=True, device=device)