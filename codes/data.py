import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from preprocessing import prepare_image

def find_image_file(directory, img_id):
    img_id = str(img_id).strip()  # Ensure no leading/trailing spaces
    for ext in ['.png', '.jpeg', '.jpg']:
        img_name = os.path.join(directory, img_id + ext)
        if os.path.exists(img_name):  # Check if the file exists
            return img_name
    raise FileNotFoundError(f"No image found for {img_id} with any known extension in {directory}")
class EyeData(Dataset):
    def __init__(self, data, directory, transform=None, image_size=256):
        self.data = data
        self.directory = directory
        self.transform = transform
        self.image_size = image_size  # Store image size
 def __len__(self):
        return len(self.data)  # Return the number of samples

    def __getitem__(self, idx):
        img_id = str(self.data.loc[idx, 'id_code']).strip()  # Get image ID
        img_name = find_image_file(self.directory, img_id)  # Find image file
        image = prepare_image(img_name, self.image_size, do_random_crop=True)  # Preprocess the image
        if self.transform is not None:
            image = self.transform(image)  # Apply transformations if any
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])  # Get label
        return {'image': image, 'label': label}  # Return the image and label
