
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# import general libraries
import os
import numpy as np

from PIL import Image

np.int = int

# model parameters
batch_size = 1
input_height = 512
input_width = 512


# create the data transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(input_height, input_width),
                                antialias=True)])

# define the dataset
class dehazeDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, transform=transform):
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.raw_images = []
        self.ref_images = []

        for filename in os.listdir(raw_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.raw_images.append(os.path.join(raw_dir, filename))
                ref_filename = filename
                self.ref_images.append(os.path.join(ref_dir, ref_filename))

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_image_path = self.raw_images[idx]
        ref_image_path = self.ref_images[idx]
        raw_image = Image.open(raw_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            ref_image = self.transform(ref_image)

        return {'raw_image': raw_image, 'ref_image': ref_image, 'filename': os.path.basename(raw_image_path)}


