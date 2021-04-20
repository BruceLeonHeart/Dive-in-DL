#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
def corr2d(X, K): # 本函数已保存在d2lzh_pytorch包中方方便便以后使用用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# In[2]:


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
corr2d(X, K)


# In[3]:


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# In[4]:


X = torch.ones(6, 8)
X[:, 2:6] = 0
X


# In[5]:


K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
Y


# In[8]:


conv2d = Conv2D(kernel_size=(1, 2))
step = 50
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))


# In[9]:


print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)


# In[10]:


import torch
from torch import nn
# 定义一一个函数来计算卷积层。它对输入入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量量大大小小和通道数(“多输入入通道和多输出通道”一一节将介绍)均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) # 排除不不关心心的前两维:批量量和通道
# 注意这里里里是两侧分别填充1行行行或列列,所以在两侧一一共填充2行行行或列列
conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=3,padding=1)

X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape


# In[11]:


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape


# In[12]:


conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=
(3, 4))
comp_conv2d(conv2d, X).shape


# 多输入通道和多输出通道

# In[13]:


import torch
from torch import nn


# In[16]:


def corr2d_multi_in(X, K):
    # 沿着X和K的第0维(通道维)分别计算再相加
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


# In[17]:


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
corr2d_multi_in(X, K)


# In[18]:


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历,每次同输入入X做互相关计算。所有结果使用用stack函数合并在一一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# In[19]:


K = torch.stack([K, K + 1, K + 2])
K.shape # torch.Size([3, 2, 2, 2])


# In[20]:


corr2d_multi_in_out(X, K)


# 1 X 1卷积

# In[21]:


def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i,h*w)
    K = K.view(c_o,c_i)
    Y = torch.mm(K,X)
    return Y.view(c_o,h,w)


# In[22]:


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
(Y1 - Y2).norm().item() < 1e-6


# POOL

# In[23]:


import torch
from torch import nn
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# In[24]:


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))


# In[25]:


pool2d(X, (2, 2), 'avg')


# 填充和步幅

# In[26]:


X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
X


# In[27]:


pool2d = nn.MaxPool2d(3)
pool2d(X)


# In[28]:


pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)


# In[29]:


pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
pool2d(X)


# In[30]:


X = torch.cat((X, X + 1), dim=1)
X


# LeNet

# In[50]:


import time
import torch
from torch import nn,optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    
    def forward(self,img):
        #print("img",img.shape)
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output


# In[51]:


net = LeNet()
print(net)


# In[37]:


import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
batch_size = 256
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
train=False, download=False, transform=transforms.ToTensor())
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[48]:


def train(net,train_iter,test_iter,batch_size,
          optimizer,device,num_epoches):
    net = net.to(device)
    print("traning on ",device)
    loss = torch.nn.CrossEntropyLoss()
    batch_cnt = 0
    for epoch in range(num_epoches):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_cnt +=1
        test_acc = eval_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / batch_cnt,
                 train_acc_sum / n, test_acc, time.time() - start))


# In[39]:


def eval_accuracy(data_iter,net,device='cuda'):
    acc_sum,n = 0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(net,torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) 
                            ==y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X,is_training=False).argmax(dim=1)
                                == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) ==
                                y).float().sum().item()
            
            n += y.shape[0]
        return acc_sum/n


# In[52]:


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer, device,num_epochs)


# AlexNet

# In[60]:


import time
import torch
from torch import nn,optim
import torchvision

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
        
    def forward(self,img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))
        return output


# In[61]:


net = AlexNet()
print(net)


# In[55]:


def load_data_fashion_mnist(batch_size, resize=None,
                            root='./Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
        trans.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False, download=False, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter,test_iter

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size,resize=224)


# In[62]:


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer,device, num_epochs)


# VGG

# In[70]:


import torch
from torch import nn,optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里里里会使宽高高减半
    return nn.Sequential(*blk)


# In[64]:


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512),(2, 512, 512))
# 经过5个vgg_block, 宽高高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意


# In[67]:


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# In[77]:


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一一个vgg_block都会使宽高高减半
        net.add_module("vgg_block_" + str(i+1),
        vgg_block(num_convs, in_channels, out_channels))
        # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                            nn.Linear(fc_features,fc_hidden_units),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(fc_hidden_units,fc_hidden_units),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(fc_hidden_units, 10)
                                          ))
    return net


# In[78]:


net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)


# In[79]:


ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio),
                   (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio),
                   (2, 512//ratio,512//ratio)]
net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units //ratio)
print(net)


# In[80]:


batch_size = 64
# 如出现“out of memory”的报错信息,可减小小batch_size或resize
train_iter, test_iter = load_data_fashion_mnist(batch_size,resize=224)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, train_iter, test_iter, batch_size, optimizer,device, num_epochs)


# NIN

# 网络中的网络 
#  1. LeNet、AlexNet和VGG在设计上的共同之处是:先以由卷积层构成的模块充分抽取空间
# 特征,再以由全连接层构成的模块来输出分类结果。
#  2. 串联多个由卷积层和全连接层构成的小网络来构建一个深层网络。
