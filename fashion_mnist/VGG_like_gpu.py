import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

class VGG_like(nn.Module):
    def __init__(self):
        super(VGG_like, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Dropout layer after pooling
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)  # Another Dropout layer after pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout before the final layer
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.classifier(x)
        return x

def evaluate_model(data_iter, net, device, loss_fn):
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            total_loss += l.item() * y.size(0)
            y_pred = torch.argmax(y_hat, axis=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is available() else "mps" if torch.backends.mps.is available() else "cpu")
    print(f"Using {device} device")

    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

    net = VGG_like().to(device)
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        net.train()  # Ensure the network is in training mode
        total_loss = 0
        total_number = 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            l = loss_fn(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

            total_loss += l.item() * y.size(0)
            total_number += y.size(0)

        train_loss = total_loss / total_number
        net.eval()  # Set the network to evaluation mode
        test_loss, test_accuracy = evaluate_model(test_iter, net, device, loss_fn)

        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy * 100:.2f}%')
