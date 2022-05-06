import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 1
batch_size = 10
learning_rate = 0.001

#for analyzeing loss and accuracy
val_losses, training_losses, training_acc , val_acc = [] , [] , [] , []

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class

def create_train_validation_loaders(dataset, validation_ratio, batch_size=100):
    dataloader_train,dataloader_valid = train_test_split(dataset, test_size=validation_ratio, random_state=42)
    dl_train = torch.utils.data.DataLoader(dataloader_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_valid = torch.utils.data.DataLoader(dataloader_valid, batch_size=batch_size, shuffle=False, num_workers=0)
    return dl_train, dl_valid

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader, val_loader = create_train_validation_loaders(dataset, validation_ratio = 0.15,batch_size= batch_size)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
'''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=5,stride=1,padding=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32, device = device)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size =  5, stride=1,padding=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64, device = device)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 250)
        self.fcbatchnorm2 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1((self.conv1(x)))))  
        x = self.pool(F.relu(self.batchnorm2((self.conv2(x)))))  

        x = x.view(-1, 64 * 8 * 8)            
        x = self.dropout(F.relu(self.fc1(x)))               
        x = F.relu(self.fcbatchnorm2(self.fc2(x)))              
        x = self.fc3(x)                                  # -> n, 10
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = 0.9)
n_total_steps = len(train_loader)

def forward_one_epoch(loader,optimizer,net,mode):
    losses, av_loss = [], []
    n_correct , n_samples = 0 , 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for i, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        # Backward and optimize
        if mode == Mode.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #model accuracy
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        #progress bar
        if i %20 ==0 or i == (len(loader) - 1):
            av_loss = np.mean(losses)
            progress_bar(i, len(loader),f'Mean Loss: {av_loss:.3f}')
        
        #accuracy for each class
        if mode == Mode.test:
            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                   n_class_correct[label] += 1
                n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    return av_loss , acc, n_class_correct, n_class_samples

for epoch in range(num_epochs):
    print(f'\n----------------------------Epoch {epoch + 1}-----------------------------')
    model = model.train()
    losses,acc,_,_ = forward_one_epoch(train_loader, optimizer, model, Mode.training)
    training_losses.append(losses)
    training_acc.append(acc)
    print(f' Train loss: {training_losses[-1]:.4f} ,Train accuracy: {training_acc[-1]:.4f} %')
    model = model.eval()
    with torch.no_grad():
        losses,acc,_,_ = forward_one_epoch(val_loader, optimizer, model, Mode.validation)
    val_losses.append(losses)
    val_acc.append(acc)
    print(f' Validation loss: {val_losses[-1]:.4f}, Validation accuracy: {val_acc[-1]:.4f} %')
    scheduler.step()
print('----------------------------------------------------------------')


print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

#test run
with torch.no_grad():
    losses,acc,n_class_correct,n_class_samples = forward_one_epoch(test_loader, optimizer, model, Mode.test)

    print(f'Accuracy of the network: {acc} %, Loss of the test: {losses}')

    #print accuracy of each class
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

#polt accuracy and loss graphs
plot_loss(training_losses, val_losses)
plot_accuracy(training_acc, val_acc)
