---
title: 'Matplotlib: 隐藏刻度线和刻度值'
collection: blogs
date: 2023-12-29
permalink: /blogs/2023-12-29-hidden_tick
excerpt: '隐藏figure中的坐标刻度线和刻度值的实现方法'
tags:
  - matplotlib
---

**原图**

```python
x = np.linspace(0, 3, 100)
y = np.sin(2*np.pi*x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
```

<div align='center'>
    <img src='./images/tick1.png' width="30%">
</div>

**仅隐藏刻度值**

```python
x = np.linspace(0, 3, 100)
y = np.sin(2*np.pi*x)

fig, ax = plt.subplots()
ax.plot(x, y)

# 隐藏刻度值
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.yaxis.set_major_formatter(plt.NullFormatter())

plt.show()
```

<div align='center'>
    <img src='./images/tick2.png' width="30%">
</div>

**隐藏刻度线+刻度值**

```python
x = np.linspace(0, 3, 100)
y = np.sin(2*np.pi*x)

fig, ax = plt.subplots()
ax.plot(x, y)

# 同时隐藏tick_line和tick_value
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_locator(plt.NullLocator())

plt.show()
```

<div align='center'>
    <img src='./images/tick3.png' width="30%">
</div>



**参考：**https://www.zhihu.com/question/506717024/answer/2654103319

