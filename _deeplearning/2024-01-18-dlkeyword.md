---
title: '一网打尽深度学习中常见的英文词汇含义与用途'
date: 2024-01-18
permalink: /deeplearning/2024-01-18-dlkeyword
excerpt: '介绍pipeline、baseline、backbone、benchmark、Framework、structure等经典单词的含义和用途。'
tags:
  - 名词解释
---

## 类别1

+ **pipeline (管线)：** 指的是**一系列数据处理步骤**或任务的集合。eg: 在机器学习项目中，pipeline可能包括数据收集、清洗、特征提取和模型训练等步骤。

+ **Framework (框架)：**指的是为解决一类特定问题而设计的预制结构或方法集合。
+ **structure （结构）：**深度学习模型的**网络结构**。
+ **Architecture （架构）：**

## 类别2

* **baseline （基线）：**baseline通常是一种**传统方法**，用于验证新方法的有效性。如果新方法在某些性能指标上优于baseline，那么这表明新方法有所改进。

* **benchmark （基准）：**benchmark可能是一组**标准数据集**，用于测试和比较不同方法的性能。

## 类别3

* **Backbone （主干网络）：**backbone 是模型的主要组成部分，如卷积神经网络（CNN）或残差神经网络（ResNet）等， **用于特征提取**，以便后续的处理和分析。
* **Neck （颈部）：**介于backbone和head之间的网络，一般指**中间层**，用于对来自backbone的输出进行降维或者调整。
* **Head（头部）：**指模型的输出层，对来自neck处理后的特征进行输出。对于分类任务，则head可能是一些系列全连接层，并输出最终分类结果。

## 其他

* **warmup （热身）：**在模型训练时前面几个epoch用小的学习率热身，有助于收敛和提升训练的稳定性。一般而言，先用小的学习率热身，然后增大学习率，最后学习率衰减。
* **Bottleneck Layer（瓶颈层）：**如ResNet中，对前一个输入先对通道数降维，然后恢复通道数与输入一致，这种类似瓶颈的结构，称为瓶颈层。
* **Ground truth （真实值）：**如图像分类中， 真实的标签称为ground truth。

**参考**

[1] GPT-4

[2] [深度学习backbone、neck 和 head介绍 ](https://zhuanlan.zhihu.com/p/607578342)

[3] [面向小白的深度学习论文术语](https://zhuanlan.zhihu.com/p/138068036)

[4] [README - AITD (gitbook.io)](https://jiqizhixin.gitbook.io/artificial-intelligence-terminology-database/)
