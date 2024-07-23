import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizationLayer(nn.Module):
    def __init__(self, k, P):
        super(NormalizationLayer, self).__init__()
        self.k = k
        self.P = P

    def forward(self, x):
        # Compute the power of the input
        power = torch.mean(x ** 2)
        # Scale the input to have the desired power
        scale = torch.sqrt(self.P / (self.k * power))
        return x * scale

class AWGNLayer(nn.Module):
    def __init__(self, snr_db):
        super(AWGNLayer, self).__init__()
        self.snr_db = snr_db

    def forward(self, x):
        snr = 10 ** (self.snr_db / 10)
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / snr
        noise = torch.sqrt(noise_power) * torch.randn_like(x)
        return x + noise

# Define the model
class JSCCNet(nn.Module):
    def __init__(self, P, snr_db, input_channels=3, output_channels=3, wireless_channel=19):
        super(JSCCNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=2)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(32, 19, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(19, wireless_channel, kernel_size=5, stride=1, padding=2)
        self.prelu5 = nn.PReLU()

        # Normalization and AWGN
        self.normalization = NormalizationLayer(k=wireless_channel*5*5, P=P)
        self.awgn = AWGNLayer(snr_db)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(wireless_channel, 19, kernel_size=5, stride=1, padding=2)
        self.prelu6 = nn.PReLU()
        self.deconv2 = nn.ConvTranspose2d(19, 32, kernel_size=5, stride=1, padding=2)
        self.prelu7 = nn.PReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.prelu8 = nn.PReLU()
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.prelu9 = nn.PReLU()
        self.deconv5 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))

        # Normalization and AWGN
        x = self.normalization(x)
        x = self.awgn(x)

        # Decoder
        x = self.prelu6(self.deconv1(x))
        x = self.prelu7(self.deconv2(x))
        x = self.prelu8(self.deconv3(x))
        x = self.prelu9(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))
        return x

# Hyperparameters
batch_size = 64
learning_rate = 1e-2
num_epochs = 100
P = 1.0
snr_db = 20 
wireless_channel = 19

# Instantiate the model and move it to the GPU if available
model = JSCCNet(wireless_channel=wireless_channel, P=P, snr_db=snr_db).to(device)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Choose the dataset
if True:
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

else:
    train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.STL10(root='./data', split='test', transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Example training loop
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

if True and __name__ == '__main__':
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)  # Assuming autoencoder structure for demonstration
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")


    # Save the model
    torch.save(model.state_dict(), "JSCCNet.pth")

# Load the model checkpoint
model.load_state_dict(torch.load('JSCCNet.pth'))
model.eval()

# Function to display images
def imshow(img, title=None):
    #img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)

# Get some random test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Print images
print('Original Images')
imshow(utils.make_grid(images), 'Oringal Images')
plt.show(block=False)
plt.pause(0.1)  # Pause to display the image

images = images.to(device)
# Reconstructed images
output = model(images)
output = output.cpu()

print('Reconstructed Images')
plt.figure()
imshow(utils.make_grid(output.detach()), 'Reconstructed Images')
plt.show()