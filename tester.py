import torch
from data import IonDataset
from torch.utils.data import DataLoader
from model import IonPredictor
from util import make_valid_step, valid_loss_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './ion_data'

batch_size=256
test_data = IonDataset(data_dir, 'test')
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = IonPredictor().to(device)

checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_step = make_valid_step(model, valid_loss_fn)

with torch.no_grad():
    test_MSE_loss=0
    cnt = 0
    for x_test_batch, y_test_batch_ion, y_test_batch_potential in test_dataloader:
        # 데이터 로더에서 받은 미니배치를 device 에 업로드
        x_test_batch = x_test_batch.to(device)
        y_test_batch_ion = y_test_batch_ion.to(device)
        y_test_batch_potential = y_test_batch_potential.to(device)
        
        # 미니매치 데이터를 이용해 performance 평가
        _, eval_test_loss_regressor, correct_cnt = test_step(x_test_batch, [y_test_batch_ion, y_test_batch_potential])
        test_MSE_loss += eval_test_loss_regressor
        cnt += correct_cnt
    test_loss = test_MSE_loss / len(test_data)
    cnt = 100 * cnt / len(test_data)
    print("Test MSE loss:{}, Accuracy: {}%".format(test_loss, cnt))
    