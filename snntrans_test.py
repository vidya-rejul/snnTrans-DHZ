# SNN libraries
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# torch libraries
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import general libraries
import os
import numpy as np
import random
import tqdm


from torchmetrics.functional import structural_similarity_index_measure as ssim

from torchvision import utils as c_utils

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

# neuron parameters
tau=2.0
v_reset=0.0
v_threshold=0.25
surro_grad=surrogate.Sigmoid(alpha=4.0)
timesteps=4

from snntrans_model import dhz_net
from dataset_rgblab import dehazeDataset

# Testing
def testing(network,testloader,timesteps):
    network = network.eval()
    test_loss_hist = []
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            raw_image = data['raw_image'].to(device)
            ref_image = data['ref_image'].to(device)

            recon_img=network(raw_image)

            functional.reset_net(network)

            recon_filename = data['filename']
            for i in range(recon_img.size(0)):
                image_name = os.path.join('./dehazed_images/', recon_filename[i])
                c_utils.save_image(recon_img[i], image_name)


# train & validation dataset
test_dataset = dehazeDataset(raw_dir='./data/test/raw/',
                            ref_dir='./data/test/reference/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

net.load_state_dict(torch.load('./checkpoints/best_snntrans_100.00.pth'))
test_loss = testing(net, test_loader, timesteps)
