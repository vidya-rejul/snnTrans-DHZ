# SNN libraries
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import general libraries
import os
import numpy as np
import random
import tqdm

from torchmetrics.functional import structural_similarity_index_measure as ssim

from snntrans_model import dhz_net
from dataset_rgblab import dehazeDataset

np.int = int

# fix the random seed
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

dtype = torch.float
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# model parameters
batch_size = 1
input_height = 512
input_width = 512

timesteps=4

epochs=10
max_epoch = epochs

# Loss function
def TV_loss(recon_img):
    batch_size, channels, height, width = recon_img.shape

    tv_h = torch.pow(recon_img[:, :, 1:, :] - recon_img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(recon_img[:, :, :, 1:] - recon_img[:, :, :, :-1], 2).sum()
    tv_t = tv_h + tv_w
    tv_loss = tv_t/ (batch_size * channels * height * width)
    return tv_loss

# Train and validation
def train(network,trainloader,optimizer,timesteps):
    network = network.train()
    train_loss_hist = []
    for data in tqdm.tqdm(trainloader):
        raw_image = data['raw_image'].to(device)
        ref_image = data['ref_image'].to(device)

        optimizer.zero_grad()
        recon_img=network(raw_image)

        loss_train = F.mse_loss(recon_img, ref_image) + 0.5*(1-ssim(recon_img, ref_image, data_range=1.0)) + 0.25*TV_loss(recon_img)
        loss_train.backward()
        optimizer.step()
        functional.reset_net(network)
        train_loss_hist.append(loss_train.detach().cpu().numpy())
    return np.mean(train_loss_hist)

def validate(network,valloader,timesteps):
    network = network.eval()
    val_loss_hist = []
    with torch.no_grad():
        for data in tqdm.tqdm(valloader):
            raw_image = data['raw_image'].to(device)
            ref_image = data['ref_image'].to(device)

            recon_img=network(raw_image)
            loss_val = F.mse_loss(recon_img, ref_image) + 0.5*(1-ssim(recon_img, ref_image, data_range=1.0)) + 0.25*TV_loss(recon_img)
            functional.reset_net(network)
            val_loss_hist.append(loss_val.detach().cpu().numpy())
    return np.mean(val_loss_hist)

# train & validation dataset
train_dataset = dehazeDataset(raw_dir='./data/train/raw/',
                              ref_dir='./data/train/reference/')
val_dataset = dehazeDataset(raw_dir='./data/val/raw/',
                            ref_dir='./data/val/reference/')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# checkpoint directory
checkpoints_dir = './checkpoints/'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return trainable_params, non_trainable_params


net = dhz_net(timesteps)
net = net.to(device)

trainable_params, non_trainable_params = count_parameters(net)
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-Trainable Parameters: {non_trainable_params}")
print(f"Total Parameters: {(trainable_params + non_trainable_params) / 1e6} M")

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# Training
training_loss = []
validation_loss = []
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{max_epoch}')
    train_loss = train(net, train_loader, optimizer,timesteps)
    print(f'Train[{epoch + 1}/{max_epoch}] Reconstruction_Loss: {train_loss}')
    training_loss.append([epoch, train_loss])

    # Save the results to a text file
    with open('./checkpoints/training_loss.txt', 'w') as file:
        for loss_value in training_loss:
            file.write(f'{loss_value[0] + 1} {loss_value[1]}\n')

    if epoch % 5== 0:
        torch.save(net.state_dict(), f'{checkpoints_dir}/{epoch + 1:003}.pth')
        message = f'Epoch {epoch+1} training finished, reconstruction_loss : {train_loss :.6f}, model saved'
        print(message)

    if epoch >=2:
        print('Evaluating the model')
        val_loss = validate(net, val_loader,timesteps)
        print('Reconstruction loss (validation)= ', val_loss)
        validation_loss.append([epoch, val_loss])

        # Save the results to a text file
        with open('./checkpoints/validation_loss.txt', 'w') as file:
            for loss_value in validation_loss:
                file.write(f'{loss_value[0] + 1} {loss_value[1]}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), f'{checkpoints_dir}/best_snntrans_{best_epoch + 1:.2f}.pth')
message = f'Training finished, Best model at Epoch: {best_epoch + 1}, Best reconstruction loss: {best_val_loss:.4f}'
print(message)


