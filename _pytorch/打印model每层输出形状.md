---
title: '打印model每层输出形状的几种方法'
excerpt: '有时候自己设计模型时可能需要试凑上一层输出形状大小，这时就需要打印前面层输出大小了。'
permalink: /pytorch/打印model每层输出形状
date: 2023-10-15
tags:
  - output shape
  - hook
---

## 背景

有时候自己设计模型时可能需要试凑上一层输出形状大小，这时就需要打印前面层输出大小了。

## 案例

**注意：**如果为了试凑模型的设计参数，需要去掉`output=model(input)`，因此这样无法无法打印前面层的形状，直接全部报错。

去掉之后，会依次打印前面的形状，直到报错出终止。

### 案例1：

```python
model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), 
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), 
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120), 
        nn.Sigmoid(),
        nn.Linear(120, 84), 
        nn.Sigmoid(),
        nn.Linear(84, 10)
)

x = torch.rand(10, 1, 28, 28)
# y = model(x) # 去掉
for layer in model:
    x = layer(x)
    print(layer.__class__.__name__, ':\t', x.shape)
```

输出结果：

```python
Conv2d :	 torch.Size([10, 6, 28, 28])
Sigmoid :	 torch.Size([10, 6, 28, 28])
AvgPool2d :	 torch.Size([10, 6, 14, 14])
Conv2d :	 torch.Size([10, 16, 10, 10])
Sigmoid :	 torch.Size([10, 16, 10, 10])
AvgPool2d :	 torch.Size([10, 16, 5, 5])
Flatten :	 torch.Size([10, 400])
Linear :	 torch.Size([10, 120])
Sigmoid :	 torch.Size([10, 120])
Linear :	 torch.Size([10, 84])
Sigmoid :	 torch.Size([10, 84])
Linear :	 torch.Size([10, 10])
```

### 案例2：

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), 
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120), 
            nn.Sigmoid(),
            nn.Linear(120, 84), 
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x, *args, **kwargs):
        x = self.block(x)
        return x

cnn = CNN()
x = torch.rand(10, 1, 28, 28)
# y = model(x) # 去掉
for layer in cnn.block:
    x = layer(x)
    print(layer.__class__.__name__, ':\t', x.shape)
```

输出结果：

```python
Conv2d :	 torch.Size([10, 6, 28, 28])
Sigmoid :	 torch.Size([10, 6, 28, 28])
AvgPool2d :	 torch.Size([10, 6, 14, 14])
Conv2d :	 torch.Size([10, 16, 10, 10])
Sigmoid :	 torch.Size([10, 16, 10, 10])
AvgPool2d :	 torch.Size([10, 16, 5, 5])
Flatten :	 torch.Size([10, 400])
Linear :	 torch.Size([10, 120])
Sigmoid :	 torch.Size([10, 120])
Linear :	 torch.Size([10, 84])
Sigmoid :	 torch.Size([10, 84])
Linear :	 torch.Size([10, 10])
```

**注意：**由于`cnn`模型中包含子模块`block`，所以需要索引到对应的子模块`cnn.block`，可以从打印`print(cnn)`查看模型的层次结构：

```python
CNN(
  (block): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): Sigmoid()
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Sigmoid()
    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=400, out_features=120, bias=True)
    (8): Sigmoid()
    (9): Linear(in_features=120, out_features=84, bias=True)
    (10): Sigmoid()
    (11): Linear(in_features=84, out_features=10, bias=True)
  )
)
```

### 案例3：

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        x = self.block(x)
        x = self.fc(x)
        return x
```

**方法1：使用钩子**

```python
x = torch.randn(10, 3, 28, 28)
model = Net()

def print_size(module, input, output):
    print(module.__class__.__name__, ':\t', output.shape)

for layer in model.children():
    layer.register_forward_hook(print_size)

out = model(x)
```

输出结果：

```python
Sequential :	 torch.Size([10, 84])
Linear :	 torch.Size([10, 10])
```

> 上述方法直接打印每个block的最终输出结果。

也可以直接打印block每一层的输出形状大小`model.block.children()`：

```python
x = torch.randn(10, 3, 28, 28)
model = Net()

def print_size(module, input, output):
    print(module.__class__.__name__, ':\t', output.shape)

for layer in model.block.children():
    layer.register_forward_hook(print_size)

out = model(x)
```

输出结果：

```python
Conv2d :	 torch.Size([10, 6, 24, 24])
ReLU :	 torch.Size([10, 6, 24, 24])
MaxPool2d :	 torch.Size([10, 6, 12, 12])
Conv2d :	 torch.Size([10, 16, 8, 8])
ReLU :	 torch.Size([10, 16, 8, 8])
MaxPool2d :	 torch.Size([10, 16, 4, 4])
Flatten :	 torch.Size([10, 256])
Linear :	 torch.Size([10, 120])
ReLU :	 torch.Size([10, 120])
Linear :	 torch.Size([10, 84])
ReLU :	 torch.Size([10, 84])
```

**方法2：大于任意一层的大小**

设置一个打印层的类。

```python
class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print('shape:', x.shape)
        return x
```

将`PrintLayer()`插入到任意层后面即可。

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            PrintLayer(),
            
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU()
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc = nn.Linear(84, 10)
        

    def forward(self, x):
        x = self.block(x)
        x = self.fc(x)
        return x

x = torch.randn(10, 3, 28, 28)
model = Net()
out = model(x)
```

输出结果：

```python
shape: torch.Size([10, 6, 12, 12])
```

