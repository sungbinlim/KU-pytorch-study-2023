import time
import torch
import torch.nn as nn

loss_classifier = nn.CrossEntropyLoss()
loss_regression = nn.MSELoss()

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

def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

def make_valid_step(model, loss_fn):
    
    def valid_step_fn(x, y):
        model.eval()
        y_hat = model(x)        
        model_prediction = torch.argmax(y_hat[0], axis=1)
        target = torch.argmax(y[0], axis=1)
        correct_cnt = torch.sum(model_prediction == target)

        loss_classifier, loss_regressor = loss_fn(y_hat, y)
        
        return loss_classifier.item(), loss_regressor.item(), correct_cnt.item()
    return valid_step_fn

def train_model(train_loader, valid_loader, epochs=100, checkpoint=False):
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