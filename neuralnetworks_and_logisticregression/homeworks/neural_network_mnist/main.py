# Amisha H. Somaiya 
# CSE546 HW3

# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List
import os
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem
import torchvision.datasets as datasets
from torchvision import transforms

class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
   
        # Declaring parameters for F1 model
        self.alpha0 = 1 / math.sqrt(d)
        self.alpha1 = 1 / math.sqrt(h)
        self.w0 = Parameter(Uniform(-self.alpha0, self.alpha0).
                            sample(sample_shape=torch.Size([d, h])))
        self.b0 = Parameter(Uniform(-self.alpha0, self.alpha0).
                            sample(sample_shape=torch.Size([h])))
        self.w1 = Parameter(Uniform(-self.alpha1, self.alpha1).
                            sample(sample_shape=torch.Size([h, k])))
        self.b1 = Parameter(Uniform(-self.alpha1, self.alpha1).
                            sample(sample_shape=torch.Size([k])))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
   
        # The forward pass for F1 performing W_1(sigma(W_0*x + b_0)) + b_1
        x_ret1 = torch.matmul(x, self.w0) + self.b0
        x_ret2 = relu(x_ret1)
        x_ret = torch.matmul(x_ret2, self.w1) + self.b1
        return x_ret

class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
 
        # Declaring parameters for F2 model
        self.alpha0 = 1 / math.sqrt(d)
        self.alpha01 = 1 / math.sqrt(h0)
        self.alpha1 = 1 / math.sqrt(h1)

        self.w0 = Parameter(Uniform(-self.alpha0, self.alpha0).
                            sample(sample_shape=torch.Size([d, h0])))
        self.b0 = Parameter(Uniform(-self.alpha0, self.alpha0).
                            sample(sample_shape=torch.Size([h0])))
        self.w1 = Parameter(Uniform(-self.alpha01, self.alpha01).
                            sample(sample_shape=torch.Size([h0, h1])))
        self.b1 = Parameter(Uniform(-self.alpha01, self.alpha01).
                            ample(sample_shape=torch.Size([h1])))
        self.w2 = Parameter(Uniform(-self.alpha1, self.alpha1).
                            sample(sample_shape=torch.Size([h1, k])))
        self.b2 = Parameter(Uniform(-self.alpha1, self.alpha1).
                            sample(sample_shape=torch.Size([k])))



    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
       
        # The forward pass for F2 performing
        # W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)
        x_ret1 = torch.matmul(x, self.w0) + self.b0
        x_ret2 = relu(x_ret1)
        x_ret3 = torch.matmul(x_ret2, self.w1) + self.b1
        x_ret4 = relu(x_ret3)
        x_ret = torch.matmul(x_ret4, self.w2) + self.b2

        return x_ret

@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of
     training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
  
    epochs = 32
    losses = []
    for i in range(epochs):
        loss_epoch = 0
        acc = 0

        for batch in train_loader:
            print(train_loader)
            images, labels = batch
            images, labels = images, labels
            images = images.view(-1, 784)
            optimizer.zero_grad()
            logreg = model.forward(images)
            preds = torch.argmax(logreg, 1)
            acc += torch.sum(preds == labels).item()
            loss = cross_entropy(logreg, labels)
            loss_epoch += loss.item()
            print(loss, "loss")
            print(loss_epoch, "loss_epoch")
            loss.backward()
            optimizer.step()

            print("Epoch", i)
            print("Loss:", loss_epoch / len(train_loader.dataset))
            print("Acc: ", acc / len(train_loader.dataset))
            if acc / len(train_loader.dataset) > 0.97:
                break

        losses.append(loss_epoch / len(train_loader.dataset))

    return losses

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report total number of parameters for each network
    """
   

    print(os.getcwd())
    mnist = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
    train_loader = DataLoader(mnist, batch_size= 100, shuffle = True)
   


    n = 28
    d = 784
    h = 64
    k = 10
    epochs = 32
    h0 = h1 = 32
  
    model = F1(h, d, k)
    optimizer = Adam(model.parameters(), lr=5e-3)
    tr_loss = train(model, optimizer, train_loader)
  

    plt.figure(figsize=(10, 6))
    plt.plot(torch.arange(len(tr_loss)), tr_loss, '-x',
             label="Training Loss")
  
    plt.title("Model F1: Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    model2 = F2(h0, h1, d, k)
    optimizer = Adam(model2.parameters(), lr=5e-3)
    tr_loss = train(model2, optimizer, train_loader)

    plt.figure(figsize=(10, 6))
    plt.plot(torch.arange(len(tr_loss)), tr_loss, '-x',
             label="Training Loss")
 
    plt.title("Model F2: Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


    paras1 = sum(p.numel() for p in model.parameters())
    print("Number of parameters in F1 model:", paras1)
    paras2 = sum(p.numel() for p in model2.parameters())
    print("Number of parameters in F2 model:", paras2)


if __name__ == "__main__":
    main()
