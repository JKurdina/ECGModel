import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelECG(nn.Module):
    def __init__(self):
        super(ModelECG, self).__init__()

        self.sigm = nn.Sequential(
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid())

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 124, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 124),
            nn.ReLU(),
            nn.Unflatten(1, (64, 124)),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),
            # nn.Sigmoid()
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        sigm = self.sigm(encoded)

        return sigm


def save_model(lead, top):
    class1 = torch.load('train_class1_' + top + '_' + lead + '_.pt', weights_only=True).float()
    class2 = torch.load('train_class2_' + top + '_' + lead + '_.pt', weights_only=True)[:class1.size()[0]].float()

    train_loader = DataLoader(class1, batch_size=50, shuffle=False)

    train_loader_2 = DataLoader(class2, batch_size=50, shuffle=False)

    model = ModelECG()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss()
    losses = []

    losses1 = []
    losses2 = []
    # Обучение
    for epoch in range(100):
        for i, (data_batch_1, data_batch_2) in enumerate(zip(train_loader, train_loader_2)):
            optimizer.zero_grad()
            # Обучаем модель на батче из train_loader
            sigm_1 = model(data_batch_1)

            target1 = torch.ones(data_batch_1.shape[0], 1)
            loss_sigm_1 = criterion(sigm_1, target1)
            loss_sigm_1.backward()

            # Обучаем модель на батче из train_loader_2
            sigm_2 = model(data_batch_2)
            target2 = torch.zeros(data_batch_2.shape[0], 1)
            loss_sigm_2 = criterion(sigm_2, target2)
            loss_sigm_2.backward()

            optimizer.step()

        print('Epoch:', epoch, 'Loss binary vote:', loss_sigm_1.item(), loss_sigm_2.item())
        losses1.append(loss_sigm_1.item())
        losses2.append(loss_sigm_2.item())

    torch.save(model, f"model_{top}_{lead}.pth")