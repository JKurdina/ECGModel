import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.sigm = nn.Sequential(
            nn.Linear(64, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid())

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * 40),
            nn.ReLU(),
            nn.Unflatten(1, (64, 40)),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1),
            # nn.Sigmoid()
        )


    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        sigm = self.sigm(encoded)
        decoded = decoded.squeeze(1)

        return decoded, sigm