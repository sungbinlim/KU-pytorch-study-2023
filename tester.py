import torch
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from trainer import make_valid_step, valid_loss_fn, config
from util import draw_confusion_matrix

device = config['device']
data_dir = config['data_dir']
batch_size = 512
model = config['model']().to(device)
loss_classifier = config['loss_classifier']()
loss_regression = config['loss_regression']()

test_data = IonDataset(data_dir, 'test')
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_step = make_valid_step(model, valid_loss_fn, verbose=True)

with torch.no_grad():
    test_MSE_loss = 0
    label_list = []
    pred_list = []
    for x_test_batch, y_test_batch_ion, y_test_batch_potential in test_dataloader:
        # 데이터 로더에서 받은 미니배치를 device 에 업로드
        x_test_batch = x_test_batch.to(device)
        y_test_batch_ion = y_test_batch_ion.to(device)
        y_test_batch_potential = y_test_batch_potential.to(device)
        
        # 미니매치 데이터를 이용해 performance 평가
        _, eval_test_loss_regressor, target, predict = test_step(x_test_batch, [y_test_batch_ion, y_test_batch_potential])
        label_list.append(target.numpy())
        pred_list.append(predict.numpy())
        test_MSE_loss += eval_test_loss_regressor * batch_size
    test_loss = test_MSE_loss / len(test_data)
    accuracy = draw_confusion_matrix(label_list, pred_list)
    print("Test MSE loss:{}, Accuracy: {}%".format(test_loss, accuracy))
    