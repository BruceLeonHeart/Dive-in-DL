#!/usr/bin/env python
# coding: utf-8

# # 基础

# ## 1.创建Tensor

# ### 1.1.未初始化

# In[4]:


import torch
x = torch.empty(5,3)
print(x)


# ### 1.2.随机初始化

# In[6]:


x = torch.rand(5,3)
print(x)
print(x.dtype)


# ### 1.3.全0初始化

# In[7]:


x = torch.zeros(5,3,dtype=torch.long)
print(x)


# ### 1.4.列表创建

# In[8]:


x = torch.tensor([5.5, 3])
print(x)


# In[11]:


x = x.new_ones(5, 3, dtype=torch.float64)
print(x)
# 返回的tensor默认具有相同的
x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
print(x)
print(x.size())
print(x.shape)


# | 函数             |      功能 |
# | :--- | :--- |
# | Tensor(*sizes)   |     基础构造函数 |
# | tensor(data,)    |     类似np.array的构造函数 |
# | ones(*sizes)      |    全1Tensor |
# | zeros(*sizes)      |   全0Tensor |
# | eye(*sizes)        |   对⻆角线为1,其他为0 |
# | arange(s,e,step)    |  从s到e,步⻓长为step |
# | linspace(s,e,steps)  | 从s到e,均匀切分成steps份rand/randn(*sizes) 均匀/标准分布 |
# | normal(mean,std)      | 正态分布 |
# | uniform(from,to)      | 均匀分布 |
# | randperm(m)           | 随机排列列 |

# # 2.操作

# ### 2.1.加法

# In[12]:


x = torch.ones(5,3)
y = torch.rand(5,3)
print(x+y)


# In[13]:


print(torch.add(x,y))


# In[14]:


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


# In[15]:


# adds x to y
#inplace模式
y.add_(x)
print(y)


# ### 2.2.索引

# In[16]:


#共享内存
x = torch.ones(5,3)
y = x[0,:]
y+=1
print(y)
print(x)


#  | 函数                 |                功能 |
#  | :--- | :--- |
#  | index_select(input, dim, index)   |  在指定维度dim上选取,比如选取某些行行、某些列 |
# | masked_select(input, mask)     |     例子如上,a(a>0),使用ByteTensor进行选取 |
# | non_zero(input)      |                非0元素的下标 |
# | gather(input, dim, index)     |      根据index,在dim维度上选取数据,输出的size与index一样 |

# ### 2.3.改变形状

# In[17]:


y = x.view(15)
z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
#view仅仅改变了张量的观察角度


# In[18]:


#深度拷贝
#使用clone还有一个好处是会被记录在计算图中,即梯度回传到副本时也会传到源Tensor。
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)


# In[19]:


#item()，它可以将一个标量Tensor转换成一个Python number
x = torch.randn(1)
print(x)
print(x.item())


# ### 2.4.线性代数

# | 函数          |    功能 |
# | :--- | :--- |
# | trace         |   对⻆角线元素之和(矩阵的迹) |
# | diag          |   对⻆角线元素 |
# | triu/tril     |   矩阵的上三⻆角/下三⻆角,可指定偏移量 |
# | mm/bmm        |   矩阵乘法,batch的矩阵乘法 |
# | addmm/addbmm/addmv/addr/badbmm..  | 矩阵运算 |
# | t           |     转置 |
# | dot/cross   |     内积/外积 |
# | inverse     |     求逆矩阵 |
# | svd         |     奇异值分解 |

# ## 3.广播机制

#     前面我们看到如何对两个形状相同的Tensor做按元素运算 
#     当对两个形状不同的 Tensor 按元素运算时,可能会触发广播(broadcasting)机制: 
#     先适当复制元素使这两个 Tensor 形状相同后再按元素运算

# In[20]:


x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)


# ## 4.内存开销

#     索引、 view 是不会开辟新内存的,而像 y = x + y 这样的运算是会新开内存的,然后将y指向新内存  
#     为了演示这一点,我们可以使用Python自带的 id 函数:如果两个实例的ID一致,那么它们所对应的内存地址相同;反之则不同。 

# In[21]:


x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False


# 如果想指定结果到原来的 y 的内存,我们可以使用前面介绍的索引来进行替换操作。<br>
# 在下面面的例子中,
# 我们把 x + y 的结果通过[:]写进 y 对应的内存中。

# In[22]:


x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True


# In[23]:


x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True


# ## 5.与Numpy进行转换

# 使用numpy()和from_numpy()进行高速转换，但注意：这两个函数所产生的 Tensor 和NumPy中的数组共享相同的内存

# In[26]:


a = torch.ones(5)
print("Tensor default type: " ,a.dtype)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)


# In[27]:


import numpy as np
a = np.ones(5)
print("Numpy default type: " ,a.dtype)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)


#     常用的方法就是直接用 torch.tensor() 将NumPy数组转换成 Tensor ,需要
#     注意的是该方法总是会进行数据拷贝,返回的 Tensor 和原来的数据不再共享内存。

# In[28]:


c = torch.tensor(a)
a += 1
print(a, c)


# ## 6.GPU

# In[31]:


# 以下代码只有在PyTorch GPU版本上才会执行行行
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))


# ## 7.自动求梯度

# 1.需要追踪梯度，需要指定requires_grad=True<br>
# 2.完成计算后，调用backward来完成梯度计算<br>
# 3.不想被追踪，使用detach()从记录中分离出来<br>
# 4.使用with torch.no_grad()来进行模型评估<br>

# In[32]:


x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)
#x这种直接创建的称为叶子子节点,叶子子节点对应的 grad_fn 是 None


# In[33]:


y = x + 2
print(y)
print(y.grad_fn)


# In[34]:


print(x.is_leaf, y.is_leaf)


# In[35]:


z = y * y * 3
out = z.mean()
print(z, out)


# In[36]:


a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)


# In[37]:


out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)


# In[38]:


# 再来反向传播一一次,注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)
out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)


# 1.x.grad 是和 x 同形的张量<br>
# 2.在最终节点类型不为标量的情况下，需要传输同形的权重矩阵进行反向传播

# In[39]:


x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)


# In[40]:


v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)


# In[42]:


x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True


# In[43]:


#梯度部分回传
y3.backward()
print(x.grad)


#     如果我们想要修改 tensor 的数值,但是又不希望被 autograd 记录(即不会影响反向传播),那么我们可以对 tensor.data 进行操作。

# In[44]:


x = torch.ones(1,requires_grad=True)
print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值,不会记录在计算图,所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)

