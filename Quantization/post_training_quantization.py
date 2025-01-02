import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os

MODEL_FILENAME = 'simplenet_ptq.pt'
# Define the device
device = "cpu"

def prepare_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Create a dataloader for the training
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    # Load the MNIST test set
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
    return train_loader, test_loader

def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

def test(test_loader, model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0

    iterations = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    print(f'Accuracy: {round(correct/total, 3)}')
           
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')
    
class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x
    
def main():
    # Make torch deterministic
    _ = torch.manual_seed(0)
    
    train_loader, test_loader = prepare_dataset()
    
    net = VerySimpleNet().to(device)
    
    if Path(MODEL_FILENAME).exists():
        net.load_state_dict(torch.load(MODEL_FILENAME))
        print('Loaded model from disk')
    else:
        train(train_loader, net, epochs=1)
        # Save the model to disk
        torch.save(net.state_dict(), MODEL_FILENAME)
    
    # Print the weights matrix of the model before quantization
    print('Weights before quantization')
    print(net.linear1.weight)
    print(net.linear1.weight.dtype)
    
    print('Size of the model before quantization')
    print_size_of_model(net)
    
    print(f'Accuracy of the model before quantization: ')
    test(test_loader, net)
    
    net_quantized = QuantizedVerySimpleNet().to(device)
    # Copy weights from unquantized model
    net_quantized.load_state_dict(net.state_dict())
    net_quantized.eval()

    net_quantized.qconfig = torch.ao.quantization.default_qconfig
    net_quantized = torch.ao.quantization.prepare(net_quantized) # Insert observers
    print(net_quantized)
    
    test(test_loader, net_quantized)
    
    print(f'Check statistics of the various layers')
    print(net_quantized)
    
    net_quantized = torch.ao.quantization.convert(net_quantized)
    print(f'Check statistics of the various layers')
    print(net_quantized)
    
    # Print the weights matrix of the model after quantization
    print('Weights after quantization')
    print(torch.int_repr(net_quantized.linear1.weight()))
    
    print('Original weights: ')
    print(net.linear1.weight)
    print('')
    print(f'Dequantized weights: ')
    print(torch.dequantize(net_quantized.linear1.weight()))
    print('')
    
    print('Size of the model after quantization')
    print_size_of_model(net_quantized)
    print('Testing the model after quantization')
    test(test_loader, net_quantized)
    
if __name__ == "__main__":
    main()
