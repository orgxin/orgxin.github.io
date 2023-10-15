---
title: 'TensorDataset和DataLoader用法'
date: 2023-10-15
collection: pytorch
permalink: /pytorch/TensorDataset和DataLoader用法
excerpt: 'TensorDataset、DataLoader API的用法，以及如何读取、查看dataset和dataloader。'
tags:
  - API
  - 教程
---

## 一、TensorDataset的用法

### 1.1 API用法

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 创建两个张量
data_tensor = torch.randn(5, 3, 28, 28)
target_tensor = torch.randn((5,))

# 使用TensorDataset包装这些张量
dataset = TensorDataset(data_tensor, target_tensor)
```

### 1.2 直接索引

```python
# 所有第一个样本
data_sample, target_sample = dataset[0]
```

### 1.3 使用切片

 ```python
 # 索引前三个样本
 subset_data, subset_target = dataset[:3]
 
 print(subset_data.shape, subset_target.shape)
 >>>
 torch.Size([3, 3, 28, 28]), torch.Size([3])
 ```

### 1.4 使用 `for` 循环

 ```
 # 查看TensorDataset中的内容
 for data, target in dataset:
     print(data, target)
 ```

### 1.5 转换为标准的张量

通过转换，我们可以获得原始的x_tensor和y_tensor，而不再是dataset类型。

 ```python
 # 将dataset全部转化为tensor
 data_tensor = dataset.tensors[0]
 target_tensor = dataset.tensors[1]
 
 print(data_tensor.shape, target_tensor.shape)
 >>>
 torch.Size([5, 3, 28, 28]), torch.Size([5])
 ```

### 1.6 查看长度和大小

```python
len(dataset)
>>>
5
```

## 二、DataLoader用法

### 2.1 API用法

```python
from torch.utils.data import DataLoader

trainloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

参数：

+ **batch_size (可选)：**每个批次的数据量。(default: `1`)

+ **drop_last (可选)：**如果数据集大小不能被批量大小整除，是否丢弃最后一个不完整的批次。(default: `False`)
+ **shuffle (可选)：**是否在每个 epoch 开始时打乱数据。(default: `False`)
+ **num_workers (可选)：**使用多少个子进程加载数据，加速数据加载。默认为 0，表示主进程会做所有的加载工作。
+ **pin_memory (可选)：**是否将数据加载到固定的内存中，通常用于加速数据从 CPU 到 GPU 的传输。(default: `False`)

### 2.2 使用 `iter` 和 `next`

```python
sample_batch = next(iter(dataloader))
```

**注意：**`iter`用于创建一个迭代器，`next`每次从迭代器中读取一个bach的数据，区别下面两种写法：

#### 案例1：

关于`next(iter(dataloader))`的用法。

```python
data_tensor = torch.randn(6, 3, 28, 28)
target_tensor = torch.randn((6,))
print('target_tensor:', target_tensor,'\n')

# 使用TensorDataset包装这些张量
dataset = TensorDataset(data_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for i in range(10):
    x, y = next(iter(dataloader))
    print(f'{i}:', y)
>>>
target_tensor: tensor([ 0.3627, -0.6271, -1.0919,  0.2115,  0.5562, -1.2587]) 

0: tensor([ 0.3627, -1.0919])
1: tensor([ 0.3627, -0.6271])
2: tensor([-0.6271,  0.5562])
3: tensor([0.2115, 0.3627])
4: tensor([ 0.5562, -0.6271])
5: tensor([-0.6271,  0.5562])
6: tensor([-1.0919,  0.2115])
7: tensor([-0.6271, -1.0919])
8: tensor([ 0.3627, -1.2587])
9: tensor([-0.6271,  0.5562])
```

`next(iter(dataloader))`每次都要创建一个新的迭代器，导致每次获取dataloader的第一个batch的数据，而不是继续从上次停下的地方获取，即不是完整的遍历整个dataset数据集。

#### 案例2：

关于`next(iter_dl)`的用法。

```python
iter_dl = iter(dataloader)
for i in range(10):
    x, y = next(iter_dl)
    print(y)
>>>
0: tensor([ 0.2115, -0.6271])
1: tensor([-1.0919,  0.5562])
2: tensor([ 0.3627, -1.2587])
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
```

可以发现，next遍历完dataset之后就开始报错，即这中写法会将整个dataset给读取完（不会重复读取），读取完之后就终止。

#### 案例3：

如何采用`iter`和`next`无限读取batch数据。

```python
iter_dl = iter(dataloader)
for i in range(10):
    try:
        x, y = next(iter_dl)
    except StopIteration:
        iter_dl = iter(dataloader)
        x, y = next(iter_dl)
    print(f'{i}:', y)
>>>
0: tensor([-1.2587, -0.6271])
1: tensor([0.5562, 0.2115])
2: tensor([ 0.3627, -1.0919])

3: tensor([ 0.3627, -0.6271])
4: tensor([ 0.5562, -1.2587])
5: tensor([ 0.2115, -1.0919])

6: tensor([-0.6271, -1.0919])
7: tensor([ 0.3627, -1.2587])
8: tensor([0.5562, 0.2115])

9: tensor([ 0.3627, -1.0919])
```

可以发现，读取完整个dataset之后，又开始重新读取dataset，但是每个batch是乱序的，与上次去读一个完整的dataset顺序不一致。

### 2.3 使用`for`遍历整个dataste

```python
for i, (inputs, labels) in enumerate(trainloader):
    pass
```

### 2.4 查看 DataLoader 的长度

```python
num_batches = len(dataloader)
```

### 2.5 查看 DataLoader 的属性

`DataLoader` 本身有一些属性，例如：

- `dataloader.batch_size`: 返回批处理大小。
- `dataloader.dataset`: 返回原始的数据集。
- `dataloader.num_workers`: 返回使用的工作进程数。

