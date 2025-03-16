import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=784 #28x28
hidden_size=500
num_classes=10 #10 digits
num_epochs=2
batch_size=100
learning_Rate=0.001

#MNIST dataset

train_dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

class NeuralNet(nn.Module):
  def __init__(self,input_size,hidden_size,num_classes):
    super(NeuralNet,self).__init__()
    self.l1=nn.Linear(input_size,hidden_size)
    self.relu=nn.ReLU()
    self.l2=nn.Linear(hidden_size,num_classes)

  def forward(self,x):
    out=self.l1(x)
    out=self.relu(out)
    out=self.l2(out)
    return out

#creating an instance
model = NeuralNet(input_size,hidden_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_Rate)

n_total_Steps = len(train_loader)
for epoch in range (num_epochs):
  for i,(images,labels) in enumerate(train_loader):
    images=images.reshape(-1,28*28).to(device)
    labels= labels.to(device)

    outputs = model(images)
    loss=criterion(outputs,labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
torch.save(model.state_dict(), 'mnist_model.pth')