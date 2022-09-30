from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np
from utils import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VGGNet(nn.Module):
    def __init__(self):

        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):

        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

image_size = 356
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

model = VGGNet().to(device).eval()
original_image = load_image('')
style_image = load_image('')
# generated = torch.randn(original_image.shape, device=device, requires_grad=True)
generated = original_image.clone().requires_grad_(True)
#! hyperparameters:
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)
for step in range(total_steps):
    generated_features = model(generated)
    original_image_features = model(original_image)
    style_features = model(style_image)

    style_loss = original_loss = 0
    for gen_feature, orig_feature, style_feature in zip(generated_features, original_image_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature)**2)

        #! Gram Matrix:
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G-A)**2)
    
    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(total_loss)
        save_image(generated, 'generated.png')

