import os
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 20

BATCH_SIZE  = 32
LATENT_SIZE = 64

device = torch.device('mps')

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.m1  = nn.Linear(LATENT_SIZE, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.m2  = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.m3  = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.m4  = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.m5 = nn.Linear(256, 784)

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.m1(x)))
        x = self.relu(self.bn2(self.m2(x)))
        x = self.relu(self.bn3(self.m3(x)))
        x = self.relu(self.bn4(self.m4(x)))
        x = self.tanh(self.m5(x))

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1,   64,  kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.f  = nn.Flatten()
        self.d1 = nn.Linear(256 * 4 * 4, 1)

        self.relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        x = self.drop(self.relu(self.c1(x)))
        x = self.drop(self.relu(self.c2(x)))
        x = self.drop(self.relu(self.c3(x)))

        x = self.f(x)
        x = self.d1(x)

        return x

def load_data(label):
    data = (np.load(f'data/{label}.npy').astype(np.float32) / 255) * 2 - 1

    dataset = TensorDataset(torch.tensor(data))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader

def train(label):
    os.makedirs('models', exist_ok=True)
    os.makedirs(f'out/{label}', exist_ok=True)

    gen  = Generator().to(device)
    disc = Discriminator().to(device)

    g_optim = Adam(gen.parameters(),  2e-4)
    d_optim = Adam(disc.parameters(), 1e-4)

    criterion = nn.BCEWithLogitsLoss()

    loader = load_data(label)

    for epoch in range(EPOCHS):
        print(f'epoch {epoch}')
        print('---')

        for i, (real,) in enumerate(iter(loader)):
            real = real.to(device)

            size = real.size(0)
            real_labels = torch.ones(size, 1, device=device)
            fake_labels = torch.zeros(size, 1, device=device)

            # -- train disc --
            # generate and disconnect fake data
            r = torch.randn(size, LATENT_SIZE, device=device)
            fake = gen(r).detach()

            # score data
            d_real_loss = criterion(disc(real), real_labels)
            d_fake_loss = criterion(disc(fake), fake_labels)

            d_loss = d_real_loss + d_fake_loss

            # zero
            d_optim.zero_grad()

            # backprop
            d_loss.backward()

            # step and clear
            d_optim.step()

            # -- train gen --
            # score generated data
            r = torch.randn(size, LATENT_SIZE, device=device)
            d_score = disc(gen(r))
            g_loss = criterion(d_score, real_labels)

            # zero
            g_optim.zero_grad()

            # backprop
            g_loss.backward()

            # step and clear
            g_optim.step()

            # -- logging --
            if i % 100 == 0:
                print(f'e: {epoch:02d}, i: {i:04d} | d: {d_loss:.3f}, g: {g_loss:.3f}')

        gen.eval()

        r = torch.randn(BATCH_SIZE, LATENT_SIZE, device=device)
        ary = gen(r).cpu().detach().numpy()

        ary = ary.reshape(BATCH_SIZE, 28, 28)

        for i, data in enumerate(ary):
            img = Image.fromarray(((data + 1) / 2 * 255).astype(np.uint8))
            img.save(f'out/{label}/{epoch},{i}.png')

        gen.train()

    torch.save(gen.state_dict(), f'models/{label}.pth')

def inference(label):
    gen = Generator().to(device)

    gen.load_state_dict(torch.load(f'models/{label}.pth'))
    gen.eval()

    r = torch.randn(BATCH_SIZE, LATENT_SIZE, device=device)
    ary = gen(r).cpu().detach().numpy()

    ary = ary.reshape(BATCH_SIZE, 28, 28)
    img = Image.fromarray((ary[0] * 255).astype(np.uint8))

    img.show()

def main():
    commands = {
        'train': train,
        'eval':  inference
    }

    if len(sys.argv) == 3 and sys.argv[1] in commands:
        commands[sys.argv[1]](sys.argv[2])
    else:
        print('error: incorrect arguments')
        return 1

    return 0
