import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, prepare, convert
from tqdm import tqdm
import os

# Setting the random seed for reproducibility
torch.manual_seed(0)

# Define the transformation to apply to the images (normalization)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load the MNIST dataset
def load_data():
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True)
    return train_loader, test_loader

# Define the neural network model with Quantization support
class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet, self).__init__()
        self.quant = QuantStub()
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = DeQuantStub()

    def forward(self, img):
        x = self.quant(img)
        x = x.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x

# Training function
def train(train_loader, net, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for data in data_iterator:
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            data_iterator.set_postfix(loss=loss_sum/(num_iterations+1))

# Evaluation function
def evaluate(test_loader, net, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Function to save model weights
def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

# Function to compare file sizes
def compare_file_sizes(file1, file2):
    size1 = os.path.getsize(file1)
    size2 = os.path.getsize(file2)
    return size1, size2

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()

    # Initialize the model
    net = QuantizedVerySimpleNet().to(device)

    # Save original model weights before quantization
    original_weights_file = 'original_weights.pth'
    save_model_weights(net, original_weights_file)

    # Prepare the model for quantization
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepare(net, inplace=True)

    # Train the model with quantization aware training (QAT)
    train(train_loader, net, device, epochs=5)

    # Convert the model to a quantized version
    convert(net, inplace=True)

    # Save quantized model weights
    quantized_weights_file = 'quantized_weights.pth'
    save_model_weights(net, quantized_weights_file)

    # Evaluate accuracy with original and quantized weights
    original_accuracy = evaluate(test_loader, net, device)
    quantized_accuracy = evaluate(test_loader, net, device)

    # Compare file sizes
    original_size, quantized_size = compare_file_sizes(original_weights_file, quantized_weights_file)

    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")
    print(f"Original Weights File Size: {original_size / 1024:.2f} KB")
    print(f"Quantized Weights File Size: {quantized_size / 1024:.2f} KB")
    print(f"Size Reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

if __name__ == "__main__":
    main()
