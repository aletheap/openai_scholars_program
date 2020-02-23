import torch
from torch import nn
import matplotlib
import torchvision
from torchvision import transforms
import matplotlib.pyplot as pyplot


model = nn.Linear(in_features=1, out_features=1)
x = torch.tensor([[2.0], [3.3]])


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

    def forward(self, x):
        return self.conv2d(x)

class SoftMax(nn.Softmax):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

    def forward(self, x):
        return self.conv2d(x)


class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    sef forward(self, x):
        pred = self.linear(x)
        return pred

model = LR(1,1)
x = torch.tensor([[1.0], 2.0])



#resnet = torchvision.models.resnet50(pretrained=False, progress=True)
root = "/home/apower/data/oxford-iiit-pet/breeds"

# https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
transform = transforms.Compose([transforms.Resize(256),        
                                transforms.CenterCrop(224),    
                                transforms.ToTensor(),         
                                transforms.Normalize(          
                                mean=[0.485, 0.456, 0.406],    
                                std=[0.229, 0.224, 0.225]      
                                )])

# Let's try something basic like Conv2d -> Conv2d -> Sigmoid

dataset = torchvision.datasets.ImageFolder(root, transform)

def forward(batch, ):
    Conv2d


model = Linear(in_features=1, out_features=1)