# 3-channel digit recognizer   

## Library import
``` python
import torch
from torch import nn, optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import tqdm
from torch.nn import ModuleList
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchsummary as summary
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
```

## Data preparations
```python
train_data_mnist = datasets.MNIST('./datasets', train=True, download=True, transform=transforms.ToTensor())
test_data_mnist = datasets.MNIST('./datasets', train=False, download=True, transform=transforms.ToTensor())
```
Basically, images are from MNIST data.

### Data augmentation
```python
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
```
Define custom data set.
- Use openCV
  ```python
  def mnist_image_augment(mnist, scale=True, rotate=True, shear=True, colour=True, gaussian=True, invert=True):
    l = len(mnist)

    SC = np.random.normal(1, 0.3, size=l) # scale
    SH = np.random.normal(0, 1, size=(l, 3, 2)) # shear
    R = np.random.normal(0, 20, size=l) # rotate
    C = np.random.randint(12, size=l) # colour
    G = np.random.randint(30, size=l) # noise
    I = np.random.randint(2, size=l) # invert

    augmented = []
    idx = []

    for i, t in enumerate(mnist):
        X,y = t[0], t[1]
        # y = torch.Tensor([y])
        # y = torch.squeeze(y)
        X = X.numpy()
        X = (np.reshape(X, (28, 28, 1)) * 255).astype(np.uint8)

        if scale or rotate:
            if scale:
                sc = SC[i] if SC[i] >= 0 else -SC[i]
            else:
                sc = 1
            r = R[i] if rotate else 0

            M = cv2.getRotationMatrix2D((14, 14), r, sc)
            X = cv2.warpAffine(X, M, (28, 28))
        
        if shear:
            pts1 = np.float32([[4, 4], [4, 24], [24, 4]])
            pts2 = np.float32([[4+SH[i][0][0], 4+SH[i][0][1]], [4+SH[i][1][0], 24+SH[i][1][1]], [24+SH[i][2][0], 4+SH[i][2][1]]])
            
            M = cv2.getAffineTransform(pts1, pts2)
            X = cv2.warpAffine(X, M, (28, 28))

        if colour:
            X = cv2.applyColorMap(X, C[i])
        
        if gaussian:
            g = G[i]/100 if G[i] > 0 else - G[i]/100
            gauss = np.random.normal(0, g**0.5, X.shape)
            X = (X + gauss).astype(np.uint8)

        if invert:
            X = cv2.bitwise_not(X)

        recover = (np.reshape(X, (3, 28, 28)) / 255).astype(np.float32)   
        X = torch.from_numpy(recover)
        augmented.append(X)
        idx.append(y)
    
    return augmented, idx
  ```
  With openCV library, apply various augmentation like ```scale, rotate, shear, colour, gaussian, invert``` on MNIST data.   
  ```python
  augmented, idx = mnist_image_augment(train_data_mnist)
  augmented_test, idx_test = mnist_image_augment(test_data_mnist)
  c_train_set, c_val_set = torch.utils.data.random_split(augmented, [50000, 10000])
  
  c_train_loader = torch.utils.data.DataLoader(c_train_set, batch_size=batch_size, shuffle=True)
  c_dev_loader = torch.utils.data.DataLoader(c_val_set, batch_size=batch_size)
  c_test_loader = torch.utils.data.DataLoader(augmented_test, batch_size=batch_size)
  ```
  Split data into train, validation, test set and data loader.   
  
  ref: https://colab.research.google.com/drive/14ubKqT2RunZI5FundxzYfx5QypsVrxsF
- Stanford, The Street View House Numbers (SVHN) Dataset
  ```python
  def swap_and_crop(x):
    data = np.transpose(x, (3,2,0,1))
    data = torch.Tensor(data)
    data = nn.functional.interpolate(data, size=(28,28))
    
    return data

  def label_to_tensor(y):
    data = torch.Tensor(y)
    data = torch.squeeze(data)
    data[data==10] = 0
    
    return data.type(torch.long)

  train_file = loadmat('/content/drive/MyDrive/train_32x32.mat')
  test_file = loadmat('/content/drive/MyDrive/test_32x32.mat')

  svhn_train_x = swap_and_crop(train_file['X'])
  svhn_train_y = label_to_tensor(train_file['y'])

  svhn_test_x = swap_and_crop(test_file['X'])
  svhn_test_y = label_to_tensor(test_file['y'])

  train_data_SVHN = CustomDataset(svhn_train_x, svhn_train_y)
  test_data_SVHN = CustomDataset(svhn_test_x, svhn_test_y)

  svhn_train_set, svhn_val_set = torch.utils.data.random_split(train_data_SVHN, [63257,10000])

  svhn_train_loader = DataLoader(svhn_train_set, batch_size=batch_size, shuffle=True)
  svhn_dev_loader = DataLoader(svhn_val_set, batch_size=batch_size)
  svhn_test_loader = DataLoader(test_data_SVHN, batch_size=batch_size)
  ```
  Load svhn data and resize, change dimension ```(Width, Height, Channels, Batch size)``` to ```(Batch size, Channels, Width, Height)```   
  ref: http://ufldl.stanford.edu/housenumbers/
- Others
  - [Albumentations](https://github.com/albumentations-team/albumentations)
  - Add pattern background behind MNIST image.

## Model
```python
learning_rate = 1e-3
batch_size = 64
```
Set learning rate & batch size   

```python
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
        )

        self.linear1 = nn.Linear(128, 80, bias=True)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.fc1 = nn.Sequential(
            self.linear1,
            nn.BatchNorm1d(80),
            nn.ReLU()
        )

        self.linear2 = nn.Linear(80, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.fc2 = nn.Sequential(
            self.linear2,
            nn.BatchNorm1d(10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

Model is consists of 3 convolution layers and 2 fully connected layers.   
- Convolution layer   
    - ReLU activation function
    - Batch normalization
    - Dropout
    - Average Pooling
- Fully connected layer
    - Linear function
    - Xavier initialization
    - ReLU activation function
    - Batch normalization
```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
- Loss function: Cross Entropy
- Optimizer: Adam   

## Train
```python
training_epochs = 50

for epoch in range(training_epochs):
    model.train()
    cost = 0
    n_batches = 0
    for X, Y in tqdm.tqdm(concat_train_loader, position=0, leave=True):
      if use_cuda:
          X = X.cuda()
          Y = Y.cuda()
      optimizer.zero_grad()
      y_hat = model(X)
      loss = criterion(y_hat, Y)
      loss.backward()
      optimizer.step()
      
      cost += loss.item()
      n_batches += 1
    
    cost /= n_batches
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch+1, cost))
    print("Dev")
    test(concat_dev_loader, model)
```
## Test
```python
def test(data_loader, model):
    model.eval()
    n_predict = 0
    n_correct = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(data_loader, position=0, leave=True):
            if use_cuda:
              X = X.cuda()
              Y = Y.cuda()
            y_hat = model(X)
            torch.argmax(y_hat)
            
            _, predicted = torch.max(y_hat, 1)
            
            n_predict += len(predicted)
            n_correct += (Y == predicted).sum()
        
        accuracy = n_correct/n_predict
        print(f"Accuracy: {accuracy} ()")
```
