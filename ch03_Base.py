#!/usr/bin/env python
# coding: utf-8

# # 深度学习基础

# ## 1.线性回归

# 1.模型定义<br>
# 2.模型训练<br>
# 2.1 训练数据<br>
# 2.2 损失函数<br>
# 2.3 优化算法<br>
# 3.模型预测

# ## 2.线性回归从零开始

# ### 2.1.生成数据集

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


# In[16]:


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples,
num_inputs))).float()
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01,size=labels.size())).float()


# In[17]:


print(features[0], labels[0])


# In[5]:


def use_svg_display():
    # 用用矢矢量量图显示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸寸
    plt.rcParams['figure.figsize'] = figsize


# In[7]:


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);


# ### 2.2 读取数据

# In[8]:


# 本函数已保存在d2lzh包中方方便便以后使用用
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size,
                                            num_examples)]) # 最后一一次可能不不足足一一个batch
        yield features.index_select(0, j), labels.index_select(0,j)


# In[9]:


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break


# ### 2.3初始化模型参数

# In[10]:


w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),
dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# ### 2.4定义模型

# In[11]:


def linreg(X, w, b):
    return torch.mm(X, w) + b


# ### 2.5定义损失函数

# 需要把真实值 y 变形成预测值 y_hat 的形状

# In[12]:


def squared_loss(y_hat, y): 
# 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# ### 2.6.定义优化算法

# In[13]:


def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size


# ### 2.7.训练模型

# In[19]:


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features,w,b),labels)
    print('epoch %d, loss %f '%(epoch+1,train_l.mean().item()))


# In[20]:


print(true_w, '\n', w)
print(true_b, '\n', b)


# ## 3.线性回归简洁实现

# ### 3.1.生成数据集

# In[22]:


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples,
num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,size=labels.size()), dtype=torch.float)


# ### 3.2.读取数据

# In[24]:


import torch.utils.data as Data
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小小批量量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


# In[25]:


for X, y in data_iter:
    print(X, y)
    break


# ### 3.3.定义模型

# In[27]:


from torch import nn
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
net = LinearNet(num_inputs)
print(net) # 使用用print可以打印出网网络的结构


# 1.写法一

# In[ ]:


net = nn.Sequential(nn.Linear(num_inputs,1))


# 2.写法二

# In[28]:


net = nn.Sequential()
net.add_module('linear',nn.Linear(num_inputs,1))


# 3.写法三

# In[29]:


from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))


# In[30]:


print(net)
print(net[0])


# In[32]:


for param in net.parameters():
    print(param)


# ### 3.4.初始化模型参数、定义损失、定义优化

# In[35]:


from torch.nn import init
init.normal_(net[0].weight,mean=0,std=0.01)
init.constant_(net[0].bias,val=0)
#net[0].bias.data.fill_(0)


loss = nn.MSELoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)


# ### 3.5训练模型

# In[37]:


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零,等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


# In[38]:


dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)


# ## 4.softmax回归

# 1.对比线性回归，输出为多个<br>
# 2.直接使用输出层的输出存在问题：<br>
#     2.1 由于输出层的输出值的范围不确定,我们难以直观
# 上判断这些值的意义。<br>
#     2.2 由于真实标签是离散值,这些离散值与不确定范围的输出值之间的误差难以衡量。

# softmax回归适用于分类问题。它使用softmax运算输出类别的概率分布。<br>
# softmax回归是一个单层神经网络,输出个数等于分类问题中的类别个数。<br>
# 交叉熵适合衡量两个概率分布的差异。

# ## 5.图像分类数据集

# 1. torchvision.datasets : 一些加载数据的函数及常用的数据集接口;<br>
# 2. torchvision.models : 包含常用的模型结构(含预训练模型),例如AlexNet、VGG 、
# ResNet等;<br>
# 3. torchvision.transforms : 常用的图片变换,例如裁剪、旋转等;<br>
# 4. torchvision.utils : 其他的一些有用的方法。

# In[39]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys


# In[41]:


mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST',
train=False, download=True, transform=transforms.ToTensor())


# In[42]:


print(type(mnist_train))
print(len(mnist_train), len(mnist_test))


# In[43]:


feature, label = mnist_train[0]
print(feature.shape, label) # Channel x Height X Width


# In[44]:


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress','coat','sandal', 
                   'shirt', 'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]


# In[45]:


def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# In[46]:


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


# mnist_train是torch.utils.data.Dataset的子类,所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例。

# In[59]:


batch_size = 256
print(sys.platform)
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[60]:


start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


# ## 6.softmax回归从零开始

# In[61]:


import torch
import torchvision
import numpy as np
import sys


# In[63]:


def load_data_fashion_mnist(batch_size):

    if sys.platform.startswith('win'):
        num_workers = 0 # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


# In[64]:


#获取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


# In[66]:


#初始化
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),
                 dtype=torch.float)
b = torch.zeros(num_outputs,dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# In[67]:


#多维按维度操作
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))


# In[68]:


#softmax实现
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制


# In[69]:


X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# In[70]:


#定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# In[78]:


#gather使用示例
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y.shape)
print(y.view(-1, 1))
print(y.shape)
y_hat.gather(1, y.view(-1, 1))


# In[77]:


def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))


# In[79]:


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# In[80]:


print(accuracy(y_hat,y))


# In[81]:


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# In[82]:


print(evaluate_accuracy(test_iter,net))


# In[83]:


#训练
num_epoches,lr = 5,0.1

def train(net,train_iter,test_iter,loss,num_epoches,
          batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epoches):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc))
                


# In[85]:


train(net, train_iter, test_iter, cross_entropy, num_epochs,
batch_size, [W, b], lr)


# In[86]:


X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])


# ## 7.softmax简洁实现

# In[87]:


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


# In[88]:


num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
net = LinearNet(num_inputs, num_outputs)


# In[89]:


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# In[90]:


from collections import OrderedDict
net = nn.Sequential(
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
    ('flatten', FlattenLayer()),
    ('linear', nn.Linear(num_inputs, num_outputs))])
    )


# In[91]:


init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)


# In[92]:


loss = nn.CrossEntropyLoss()


# In[93]:


optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# In[94]:


num_epochs = 5
train(net, train_iter, test_iter, loss, num_epochs,
batch_size, None, None, optimizer)


# ## 8.多层感知机

# 1.隐藏层介入：带有隐藏层的多层感知机，等价于单层<br>
# 原因是：全连接层只是对数据做仿射变换(affine transformation)，多个仿射变换的叠加仍然是一个仿射变换。解决问题的办法是引入非线性变换。

# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as np
import matplotlib.pylab as plt
import sys

def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


# In[96]:


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
print(type(x))
print(type(y))
xyplot(x, y, 'relu')


# In[97]:


y.sum().backward()
xyplot(x, x.grad, 'grad of relu')


# In[98]:


y = x.sigmoid()
xyplot(x, y, 'sigmoid')


# In[99]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')


# In[100]:


y = x.tanh()
xyplot(x, y, 'tanh')


# In[101]:


x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')


# 1.多层感知机就是含有至至少一个隐藏层的由全连接层组成的神经网络,且每个隐藏层的输出通过激活函数进行变换<br>
# 2.多层感知机是神经网络的一种

# ## 9.多层感知机从零开始

# In[102]:


import torch
import numpy as np
import sys


# In[104]:


batch_size = 256 
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs,
                                             num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,
                                             num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# In[105]:


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# In[106]:


def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# In[107]:


loss = torch.nn.CrossEntropyLoss()


# In[108]:


num_epochs, lr = 5, 100.0
train(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)


# ## 10.多层感知机简洁实现

# In[109]:


import torch
from torch import nn
from torch.nn import init
import numpy as np


# In[110]:


#模型定义
num_inputs,num_outputs,num_hiddens = 784,10,256
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)
)
for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)


# In[111]:


#数据读取与训练
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5
train(net, train_iter, test_iter, loss, num_epochs,batch_size, None, None, optimizer)


# ## 11.模型选择、欠拟合、过拟合

# 1.训练误差和泛化误差

# 2.验证数据集选择模型

# 3.K折交叉验证

# 4.数据集大小、模型复杂度对拟合的影响

# 5.多项式模拟实验

# In[112]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import numpy as np


# In[113]:


n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2),
                           torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] +
          true_w[1] *poly_features[:, 1]+
          true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01,
                                        size=labels.size()), dtype=torch.float)


# In[114]:


features[:2], poly_features[:2], labels[:2]


# In[116]:


def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,
            legend =None,figsize=(3.5,2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)


# In[117]:


num_epochs, loss = 100, torch.nn.MSELoss()
def fit_and_plot(train_features, test_features, 
                 train_labels,test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size,shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features),train_labels).item())
        test_ls.append(loss(net(test_features),test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss',test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,'\nbias:', net.bias.data)


# 正常拟合

# In[118]:


fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])


# 欠拟合

# In[119]:


fit_and_plot(features[:n_train, :], features[n_train:, :],
labels[:n_train],
labels[n_train:])


# 过拟合【样本不足】

# In[124]:


fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :],
             labels[0:2],labels[n_train:])


# ## 12.权重衰减

# 高纬线性回归实验

# In[134]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import numpy as np

n_train,n_test,num_inputs = 20,100,200
true_w,true_b = torch.ones(num_inputs,1)*0.01,0.05
features = torch.randn((n_train + n_test,num_inputs))
labels = torch.mm(features,true_w) + true_b
labels += torch.tensor(np.random.normal(0,0.01,size = labels.size()),
                      dtype=torch.float)
train_features, test_features = features[:n_train, :],features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# In[127]:


def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# In[128]:


def l2_penalty(w):
    return (w**2).sum() / 2


# In[132]:


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = linreg,squared_loss 
dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size,shuffle=True)
def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().item())
        semilogy(range(1, num_epochs + 1), train_ls, 'epochs','loss',
                 range(1, num_epochs + 1), test_ls, ['train','test'])
        print('L2 norm of w:', w.norm().item())


# In[ ]:


fit_and_plot(lambd=0)


# In[ ]:


fit_and_plot(lambd=3)


# In[137]:


def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr,
                                  weight_decay=wd) # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            # 对两个optimizer实例例分别调用用step函数,从而而分别更更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
    train_ls.append(loss(net(train_features),
    train_labels).mean().item())
    test_ls.append(loss(net(test_features),
    test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs','loss',
             range(1, num_epochs + 1), test_ls, ['train','test'])
    print('L2 norm of w:', net.weight.data.norm().item())


# In[ ]:


fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)


# ## 13.丢弃法

# In[139]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import numpy as np

def dropout(X,drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()
    return mask * X / keep_prob


# In[141]:


X = torch.arange(16).view(2,8)
dropout(X,0)


# In[142]:


dropout(X,0.5)


# In[143]:


dropout(X,1)


# In[144]:


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256,256
W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs,
                                                  num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1,
                                                  num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2,
                                                  num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]


# In[145]:


drop_prob1, drop_prob2 = 0.2, 0.5
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training: # 只在训练模型时使用用丢弃法
        H1 = dropout(H1, drop_prob1) # 在第一一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2) # 在第二二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3


# In[146]:


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数# 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1)== y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) ==y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n
                


# In[147]:


num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter =  load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)


# ## 14.正向传播、反向传播和计算图

# ## 15.数值稳定性和模型初始化

# 数值稳定性：衰减与爆炸
# 

# ## 16.房价预测

# In[ ]:





# In[ ]:




