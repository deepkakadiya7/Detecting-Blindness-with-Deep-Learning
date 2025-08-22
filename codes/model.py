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
      model.load_state_dict(torch.load(f'../models/model_{model_name}.bin', map_location=torch.device('cpu')))
        
        # Freeze first layers
        for child in list(model.children())[:-trn_layers]:
            for param in child.parameters():
                param.requires_grad = False
