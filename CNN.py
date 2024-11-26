import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

class ECG_dataset(Dataset):
    def __init__(self, ecg_signals, r_peaks):
        self.ecg_signals = ecg_signals
        self.r_peaks = r_peaks

    def __len__(self):
        return len(self.ecg_signals)

    def __getitem__(self, idx):
        # Нормализация сигнала ЭКГ
        signal = (torch.tensor(self.ecg_signals[idx], dtype=torch.float32) - torch.mean(
            torch.tensor(self.ecg_signals[idx], dtype=torch.float32))) / torch.std(
            torch.tensor(self.ecg_signals[idx], dtype=torch.float32))
        # Преобразование в тензор PyTorch
        signal = torch.tensor(signal, dtype=torch.float32)
        # Формирование входных данных (1 x 500)
        signal = signal.view(1, 500)
        # Метка - координата R-пика
        label = torch.tensor(self.r_peaks[idx], dtype=torch.float32)
        return signal, label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Входной слой
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)

        # Сверточный слой 1
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Flatten
        self.flatten = nn.Flatten()

        # Полносвязные слои
        self.fc1 = nn.Linear(in_features=64*498, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Входной слой
        # x = self.conv1(x)
        # x = nn.functional.relu(x)
        # x = self.pool1(x)
        #
        # # Сверточный слой 1
        # x = self.conv2(x)
        # x = nn.functional.relu(x)
        # x = self.pool2(x)
        #
        # # Сверточный слой 2
        # x = self.conv3(x)
        # x = nn.functional.relu(x)
        # x = self.pool3(x)
        #
        # # Flatten
        # x = self.flatten(x)
        #
        # # Полносвязные слои
        # x = self.fc1(x)
        # x = nn.functional.relu(x)
        # x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)  # Softmax для получения вероятностей
        #
        # return x
