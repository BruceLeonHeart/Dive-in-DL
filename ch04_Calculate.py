#!/usr/bin/env python
# coding: utf-8

# # 深度学习计算

# ## 模型构造

# In[1]:


import torch
from torch import nn
class MLP(nn.Module):
    # 声明带有模型参数的层,这里里里声明了了两个全连接层
    def __init__(self, **kwargs):
    # 调用用MLP父父类Block的构造函数来进行行行必要的初始化。这样在构造实例例时还可以指定其他函数
    # 参数,如“模型参数的访问、初始化和共享”一一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10) # 输出层
    # 定义模型的前向计算,即如何根据输入入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


# In[2]:


X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)


# In[4]:


class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module) # add_module方方法会将module添加进self._modules(一一个OrderedDict)
        else: # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一一个 OrderedDict,保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input


# In[5]:


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    )
print(net)
net(X)


# In[6]:


net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) #类似List的append操作
print(net[-1]) # 类似List的索引访问
print(net)


# In[19]:


net = nn.ModuleDict({
'linear': nn.Linear(784, 256),
'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)


# In[20]:


class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20),
                                      requires_grad=False) # 不不可训练参数(常数参数)
        self.linear = nn.Linear(20, 20)
        
    def forward(self, x):
        x = self.linear(x)
        # 使用用创建的常数参数,以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) +1)
        # 复用用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流,这里里里我们需要调用用item函数来返回标量量进行行行比比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# In[21]:


X = torch.rand(2, 20)
net = FancyMLP()
print(net)
net(X)


# In[22]:


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())
    def forward(self, x):
        return self.net(x)
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
net(X)


# ## 模型参数访问、初始化、共享

# In[24]:


import torch
from torch import nn
from torch.nn import init
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
#pytorch已进行行行默认初始化
print(net)
X = torch.rand(2, 4)
Y = net(X).sum()


# In[25]:


print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())


# In[26]:


print(type(net[0].named_parameters()))
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


# In[28]:


class MyModel(nn.Module):
    def __init__(self,**kwargs):
        super(MyModel,self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass
n = MyModel()
for name, param in n.named_parameters():
    print(name)


# In[29]:


weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)


# In[30]:


#参数初始化
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
    if 'bias' in name:
        init.constant_(param, val=0)
    print(name, param.data)


# 查看torch.nn.init.normal_实现<br>
# 不记录梯度
# 

# In[ ]:


def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)


# In[31]:


#自定义
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()
        
for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)


# 共享模型参数

# In[32]:


linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)


# In[33]:


print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))


# In[34]:


x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3,两次所以就是6


# ## 参数的延后初始化

# ## 自定义层

# In[36]:


import torch
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()


# In[37]:


layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))


# In[38]:


net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
y.mean().item()


# In[39]:


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(4,4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyDense()
print(net)


# In[40]:


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))})
        self.params.update({'linear3': nn.Parameter(torch.randn(4,2))}) # 新增
    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])
net = MyDictDense()
print(net)


# In[41]:


x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))


# In[43]:


net = nn.Sequential(MyDictDense(),MyDense(),)
print(net)
print(net(x))


# ## 读取与存储

# In[44]:


import torch
from torch import nn
x = torch.ones(3)
torch.save(x, 'x.pt')


# In[45]:


x2 = torch.load('x.pt')
x2


# In[46]:


y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
xy_list


# In[47]:


torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
xy


# 1.state_dict

# In[48]:


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
net = MLP()
net.state_dict()


# In[49]:


optimizer = torch.optim.SGD(net.parameters(), lr=0.001,momentum=0.9)
optimizer.state_dict()


#  ## GPU   

# In[50]:


import torch
from torch import nn
torch.cuda.is_available() # 输出 True


# In[51]:


torch.cuda.device_count()


# In[52]:


torch.cuda.current_device()


# In[53]:


torch.cuda.get_device_name(0)


# In[55]:


x = torch.tensor([1, 2, 3])
x


# In[57]:


device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
x

