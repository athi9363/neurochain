import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import onnx

# tiny CNN
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*14*14, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_one_epoch(model, device, loader, optimizer, lossfn):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = lossfn(out, target)
        loss.backward()
        optimizer.step()

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = TinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.CrossEntropyLoss()

    # tiny training: 1 epoch for demo
    train_one_epoch(model, device, train_loader, optimizer, lossfn)

    # save PyTorch
    torch.save(model.state_dict(), "models/tiny_mnist.pth")

    # export to ONNX
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy, "models/tiny_mnist.onnx", opset_version=11, input_names=['input'], output_names=['output'])
    print("Saved models/tiny_mnist.pth and models/tiny_mnist.onnx")

if __name__ == "__main__":
    main()
