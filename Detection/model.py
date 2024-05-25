import torch
import torch.nn as nn # Contain classes of training
import torch.optim as optim # Contain algorithms to train model efficassily.
import torch.nn.functional as Function
from video_dataset import VideoDataset, transform
from torch.utils.data import DataLoader

#Initializations
# Setup paths
video_dir = '/path/to/videos'
csv_file = '/path/to/labels.csv'
# Initialize dataset
dataset = VideoDataset(video_dir, csv_file=csv_file, transform=transform)
# Create data loader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

#Class defines a basic convolutional neural network with two convolutional layers followed by max-pooling and fully connected layers
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__() #Initialization of the class

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #first layer convolution layers learn local patterns in small 2D windows of the inputs.
        #3 entry for RGB (color image)
        #16 outputs 
        #kernel_size = size of small 2d window
        #stride = how far the filter moves on analysing the image(in pixels)
        #padding = prevent to hit the border of the image by adding 1 pixel more
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)# Max pooling - reducing our dimension for better analysis with taking the max value
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) # Size of 32 outputs * 56x56 image. Linear vector
        self.fc2 = nn.Linear(128, 1)  # Assuming binary classification: event or noevent, 1 class
    
    def forward(self, x):
     x = self.pool(Function.relu(self.conv1(x))) #ReLu activation to avoid negative responses
     x = self.pool(Function.relu(self.conv2(x)))
     x = x.view(-1, 32 * 56 * 56) # Convert to 1 dimension vector for input of the last layer
     x = Function.relu(self.fc1(x)) 
     x = torch.sigmoid(self.fc2(x)) #Sigmoid function to use class identification
     return x
    

#Parameters   
model = CNN() # initialization of the model
criterion = nn.BCELoss() # Rate of penalizing wrong prediction of the class
optimizer = optim.Adam(model.parameters(), lr=0.001) #Optimization model with learning rate 0.001


# Training model part
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for frames, labels in data_loader:
        if frames.nelement() == 0:
            continue

        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}')

print('Finished Training')