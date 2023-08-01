import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

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

def make_valid_step(model, loss_fn, verbose=False):
    
    def valid_step_fn(x, y):
        model.eval()
        y_hat = model(x)        
        model_prediction = torch.argmax(y_hat[0], axis=1)
        target = torch.argmax(y[0], axis=1)
        correct_cnt = torch.sum(model_prediction == target)
        loss_classifier, loss_regressor = loss_fn(y_hat, y)
        
        if verbose:
            return loss_classifier.item(), loss_regressor.item(), target.item(), model_prediction.item()
        else:
            return loss_classifier.item(), loss_regressor.item(), correct_cnt.item()
    return valid_step_fn

def draw_confusion_matrix(label_list, pred_list):
    num_classes = 4
    classes = ['5', '6', '7', '8']
    
    label_list = np.concatenate(label_list)
    pred_list = np.concagtenate(pred_list)

    confusion_matrix = metrics.confusion_matrix(label_list, pred_list, labels=[i for i in range(num_classes)])
    confusion_matrix = np.round(confusion_matrix / len(label_list), 2)

    total_correct = np.sum(pred_list == label_list)
    accuracy = total_correct / len(pred_list) * 100
    
    # Create confusion matrix plot
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Set plot labels
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - Accuracy: {round(accuracy, 3)}')
    plt.save('test_experiment.png')

    return accuracy