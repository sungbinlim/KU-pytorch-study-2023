import time
import torch

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {computation_time} seconds")
        return result
    return wrapper

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

@timer
def make_valid_step(model, loss_fn):
    def valid_step_fn(x, y):
        model.eval()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        return loss.item()
    return valid_step_fn