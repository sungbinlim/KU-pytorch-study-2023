import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MLP(nn.Module):
    def __init__(self, hidden_variables=[128, 64], input_output_dim=(1, 1)):
        super().__init__()
        self.input_variable_dim = input_output_dim[0]
        self.output_variable_dim = input_output_dim[1]
        self.list_hidden_variable = hidden_variables
        self.layer = nn.Sequential()

        variable_dim = self.input_variable_dim
        for i, hidden_variable in enumerate(self.list_hidden_variable):
            self.layer.add_module('layer_' + str(i), nn.Linear(variable_dim, hidden_variable, dtype=torch.float32))
            self.layer.add_module('activation_' + str(i), nn.ReLU())
            variable_dim = hidden_variable
        self.layer.add_module('final_layer', nn.Linear(variable_dim, self.output_variable_dim, dtype=torch.float32))

    def forward(self, x):
        y_hat = self.layer(x)
        return y_hat

class IonPredictor(nn.Module):
    def __init__(self, kernel_size=3, num_features=1000):
        super().__init__()

        self.cnn = models.resnet18()
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2, padding=3, bias=False)
        self.mlp = MLP(input_output_dim=(num_features, 5))

    def forward(self, x):

        z = self.cnn(x)
        z.view(z.size(0), -1)
        predict = self.mlp(z)
        return predict[:, :4], predict[:, 4]