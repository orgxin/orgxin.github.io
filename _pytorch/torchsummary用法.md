---
title: 'torchsummary用法'
excerpt: '一个API，用于打印模型的信息，如输入、输出大小，模型结构层次，模型的参数，每层卷积核参数等'
date: 2023-10-15
collection: pytorch
permalink: /pytorch/torchsummary用法
tags:
  - API
  - 教程
  - torchsummary
---

## 一、安装

需要使用`pip`这种方式安装：

```python
pip install torch-summary
```

导入：

```python
from torchsummary import summary
```

## 二、用法

### 2.1 常规用法

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

summary(CNN(), (1, 28, 28), col_names=["kernel_size", "input_size", "output_size", "num_params"])
```

输出：

![image-20231015212002337](./torchsummary%E7%94%A8%E6%B3%95.assets/image-20231015212002337.png)

> 注意：input shape只需要输出后三位即可，第一位batch不需要输入。

### 2.2 多输入类型

当model的forward有多个输入的时候，需要指定输入的**size**和输入的**type**。

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)

    
summary(Model(), [(1, 300), (1, 300)], dtypes=[torch.float, torch.long], col_names=["kernel_size", "input_size", "output_size", "num_params"])
```

输出：

![image-20231015212758045](./torchsummary%E7%94%A8%E6%B3%95.assets/image-20231015212758045.png)

## 三、参考文献

[torch-summary · PyPI](https://pypi.org/project/torch-summary/)