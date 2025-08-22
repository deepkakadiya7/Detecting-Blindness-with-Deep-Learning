from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
def init_model(model_name, train=True, trn_layers=2):
    '''
    Initialize the model
    '''
    # Load pre-trained model
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)

    if train:
        # Load model state dict for training mode
