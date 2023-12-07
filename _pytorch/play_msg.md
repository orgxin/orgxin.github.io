---
title: 'python弹出提示框和声音'
date: 2023-12-07
collection: pytorch
permalink: /pytorch/play_msg
excerpt: '在程序执行完成之后，希望后台可以弹出提示框，并用蜂鸣声提醒我们程序已经执行完毕。'
tags:
  - python
  - tips
---

**实现代码如下：**

```python
def play_msg():
    import tkinter as tk
    from tkinter import messagebox
    import winsound

    winsound.Beep(1000, 1200) # (Hz, s)
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("提示", "程序运行完成！")


def main():
    pass


if __name__ == "__main__":
    main()
    play_msg()
```