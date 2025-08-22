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
