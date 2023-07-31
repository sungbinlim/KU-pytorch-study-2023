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