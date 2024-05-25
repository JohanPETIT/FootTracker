import torch.nn as nn # Contain classes of training
import torch.optim as optim # Contain algorithms to train model efficassily.

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__() #Initialization of the model

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #first layer convolution layers learn local patterns in small 2D windows of the inputs.
        #kernel_size = size of small 2d window
        #stride = how far the filter moves on analysing the image(in pixels)
        #padding = prevent to hit the border of the image by adding 1 pixel more
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)  # Assuming binary classification: event or no event
