from torch import nn

class STL3DModel(nn.Module):
    def __init__(self):
        super(STL3DModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 128)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(128, 32 * 16 * 16 * 16)  # Adjust dimensions based on output size
        self.deconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = x.view(x.size(0), 32, 16, 16, 16)  # Reshape back to 3D
        x = nn.functional.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            for data in train_loader:
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")