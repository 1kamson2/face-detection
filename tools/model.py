import torch
import torch.nn as nn
torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)


class AgeModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.init_model()

    def init_model(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.out_channels, out_channels=2 * self.out_channels, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=2 * self.out_channels + 69, out_features=2 * self.out_channels),
            nn.Dropout(0.5),
            nn.Linear(2 * self.out_channels, 1),
            nn.Sigmoid(),

        )
        return model.to(self.device)

    def forward(self, model_input):
        out = self.model(model_input).unsqueeze(0)
        return out


