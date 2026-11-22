
import torch
import torch.nn as nn


class ECGAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Compress the ECG signal
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Reconstruct the original ECG signal
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16, 1, kernel_size=4, stride=2, padding=1
            )
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
