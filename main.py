import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class JSCCNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(JSCCNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=0)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(32, 19, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(19, 19, kernel_size=5, stride=1, padding=2)
        self.prelu5 = nn.PReLU()
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(19, 32, kernel_size=5, stride=1, padding=2)
        self.prelu6 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu7 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu8 = nn.PReLU()
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0)
        self.prelu9 = nn.PReLU()
        self.deconv5 = nn.ConvTranspose2d(16, output_channels, kernel_size=5, stride=2, padding=0, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))
        # Decoder
        x = self.prelu6(self.deconv1(x))
        x = self.prelu7(self.deconv2(x))
        x = self.prelu8(self.deconv3(x))
        x = self.prelu9(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))
        return x

# Instantiate the model and move it to the GPU if available
model = JSCCNet().to(device)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Example training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)  # Assuming autoencoder structure for demonstration
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")ch.git