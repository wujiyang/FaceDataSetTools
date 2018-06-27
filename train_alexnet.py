#/usr/env/bin/python
# -*- coding:utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from alexnet import *
import datetime
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

''' define a dataload class '''
rootdir = "/home/wujiyang/PyTorch/"
def default_loader(path):
    return Image.open(path)
class MyDataSet(Dataset):
    def __init__(self, path, transform = None, target_transform = None, loader = default_loader):
        fp = open(path, 'r')
        imgs = []
        for line in fp:
            pair = line.strip('\n').rstrip().split()
            imgs.append((rootdir + pair[0], int(pair[1])))
        
        self.images = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        path, label = self.images[index]
        image = self.loader(path)
        # print self.transform
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.images)
            


''' load the data for train '''
batchsize_train = 256
batchsize_test = 50
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor()])
train_data=MyDataSet(path=rootdir+'train_list.txt', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=batchsize_train, shuffle=True)
test_data=MyDataSet(path=rootdir+'val_list.txt', transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize_test, shuffle=False)


''' get the net instance '''  
net = alexnet(pretrained=True, num_classes = 100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


''' optimizer '''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma = 0.1)

''' train the network '''
for epoch in range(50): # loop over the dataset multiple times
    scheduler.step()
    # training stage
    net.train()
    training_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        #inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print statistics
        training_loss += loss.item()
        
    print('[%s, epoch %d] , training loss: %.6f, current learning rate: %.6f' % 
                  (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, training_loss / len(train_loader), scheduler.get_lr()[0]))      
    if(epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), './models/alex_model_epoch_%d.pkl' % (epoch + 1))
        
    # testing stage, test after every one epoch
    net.eval()
    correct = 0.
    total = 0.
    testing_loss = 0.
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss= criterion(outputs, labels)
            testing_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('[%s, Testing after epoch %d] Testing loss: %.6f, Test Accuracy: %.6f' % 
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, testing_loss / len(test_loader), correct / total))
       
print 'Finished Training'
torch.save(net, './models/alex_model_final.pkl')