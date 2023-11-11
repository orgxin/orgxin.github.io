---
title: 'test1'
collection: blogs
date: 2023-10-16
permalink: /blogs/test1
excerpt: '一篇文章让你搞懂分辨率、像素和尺寸大小之间的换算关系，因为plt.figure()会用到分辨率(dpi)和英寸(inch)进行保存图片。'
tags:
  - pixel
  - matplotlib
  - DPI
  - size
comments: true
tags:
  - mutual information
---

Explanation of the paper *CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information, ICML 2020*.

# 1. Introduction

This paper introduces a contrastive log-ratio upper bound of the mutual information. It provides a more stable estimation than the previously proposed L1OUT upper bound [(previous post)](https://ruihongqiu.github.io/posts/2020/09/vbmi/){:target="_blank"}.

Let's begin with this paper, [CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information](https://arxiv.org/abs/2006.12013){:target="_blank"}, in ICML 2020.

# 2. Baseline Upper Bounds

In our [previous post about variational bounds](https://ruihongqiu.github.io/posts/2020/09/vbmi/){:target="_blank"}, we discuss two types of upper bounds.

## 2.1. $$\mathrm{I}_{\mathrm{VUB}}$$

Alemi et al. (2016) introduces a variational marginal approximation $$r(y)$$ to build a variational upper bound (VUB):

$$
\begin{aligned}
\mathrm{I}(\boldsymbol{x} ; \boldsymbol{y}) &=\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}\left[\log \frac{p(\boldsymbol{y} \mid \boldsymbol{x})}{p(\boldsymbol{y})}\right] \\
&=\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}\left[\log \frac{p(\boldsymbol{y} \mid \boldsymbol{x})}{r(\boldsymbol{y})}\right]-\operatorname{KL}(p(\boldsymbol{y}) \| r(\boldsymbol{y})) \\
& \leq \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}\left[\log \frac{p(\boldsymbol{y} \mid \boldsymbol{x})}{r(\boldsymbol{y})}\right]=\mathrm{KL}(p(\boldsymbol{y} \mid \boldsymbol{x}) \| r(\boldsymbol{y})) = \mathrm{I}_{\mathrm{VUB}}.
\end{aligned}
$$

In the experiment, the variational distribution $$r(y)$$ is usually set to a standard normal distribution, which results in the estimation of MI to be high-biased.

## 2.2. $$\mathrm{I}_{\mathrm{L1Out}}$$

Poole et al. (2019) [(previous post)](https://ruihongqiu.github.io/posts/2020/09/vbmi/){:target="_blank"} replaces $$r(y)$$ with a Monte-Carlo approximation $$r_{i}(\boldsymbol{y})=\frac{1}{N-1} \sum_{j \neq i} p\left(\boldsymbol{y} \mid \boldsymbol{x}_{j}\right) \approx p(\boldsymbol{y})$$ and derives a leave one-out upper bound (L1Out):

$$
\mathrm{I}_{\mathrm{L} 1 \mathrm{Out}}:=\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N}\left[\log \frac{p\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)}{\frac{1}{N-1} \sum_{j \neq i} p\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{j}\right)}\right]\right].
$$

This bound suffers from numerical instability in practice compared with the proposed method, which will be discussed later.

## 2.3. Unknown Conditional Distribution

When the conditional distribution is unknown (not like VAE), we will introduce a neural network $$q_\theta(y\mid x)$$ to approximate $$p(y\mid x)$$ and develop variational versions of VUB and L1Out as:

$$
\mathrm{I}_{\mathrm{vVUB}}=\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}\left[\log \frac{q_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})}{r(\boldsymbol{y})}\right],
$$

$$
\mathrm{I}_{\mathrm{VL} 1 \mathrm{Out}}=\mathbb{E}\left[\frac{1}{N} \sum_{i=1}^{N}\left[\log \frac{q_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)}{\frac{1}{N-1} \sum_{j \neq i} q_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{j}\right)}\right]\right].
$$

# 3. CLUB

The full name of CLUB is Contrastive Log-ratio Upper Bound.

## 3.1. When $$p(y\mid x)$$ is Known

The bound is defined as:

$$
\mathrm{I}_{\mathrm{CLUB}}(\boldsymbol{x} ; \boldsymbol{y}):=\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]-\mathbb{E}_{p(\boldsymbol{x})} \mathbb{E}_{p(\boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]
$$

*Proof*.

Let $$\Delta$$ be the gap between the CLUB bound and the MI itself:

$$
\begin{aligned}
\Delta:=& \mathrm{I}_{\mathrm{CLUB}}(\boldsymbol{x} ; \boldsymbol{y})-\mathrm{I}(\boldsymbol{x} ; \boldsymbol{y}) \\
=& \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]-\mathbb{E}_{p(\boldsymbol{x})} \mathbb{E}_{p(\boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]-\mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})-\log p(\boldsymbol{y})]\text{, by def.}\\
=& \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[\log p(\boldsymbol{y})]-\mathbb{E}_{p(\boldsymbol{x})} \mathbb{E}_{p(\boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})],\quad\text{eliminate } \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]\\
=& \mathbb{E}_{p(\boldsymbol{y})}\left[\log p(\boldsymbol{y})-\mathbb{E}_{p(\boldsymbol{x})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})]\right],\quad \boldsymbol{x}\text{ doesn't affect }p(\boldsymbol{y}).
\end{aligned}
$$

By definition,

$$
p(\boldsymbol{y})=\int p(\boldsymbol{y} \mid \boldsymbol{x}) p(\boldsymbol{x}) \mathrm{d} \boldsymbol{x}=\mathbb{E}_{p(\boldsymbol{x})}[p(\boldsymbol{y} \mid \boldsymbol{x})].
$$

Adding $$\log$$ to both sides,

$$
\log p(\boldsymbol{y})=\log \mathbb{E}_{p(\boldsymbol{x})}[p(\boldsymbol{y} \mid \boldsymbol{x})].
$$

Remember we talked about the tractability of some bounds in the [previous post about variational bounds](https://ruihongqiu.github.io/posts/2020/09/vbmi/){:target="_blank"}, which includes the Jensen's Inequality of the expectation and $$\log$$. We apply the inequality here again,

$$
\log \left(\mathbb{E}_{p(\boldsymbol{x})}[p(\boldsymbol{y} \mid \boldsymbol{x})]\right) \geq \mathbb{E}_{p(\boldsymbol{x})}[\log p(\boldsymbol{y} \mid \boldsymbol{x})].
$$

Therefore, the gap is non-negative. CLUB is an upper bound. (Kinda) obviously, when $$\boldsymbol{x}$$ and $$\boldsymbol{x}$$ are independent, CLUB is tight.

With multiple sample pairs $$\left\{\left(\boldsymbol{x}_{i}, \boldsymbol{y}_{i}\right)\right\}_{i=1}^{N}$$, $$\mathrm{I}_{\mathrm{CLUB}}(\boldsymbol{x} ; \boldsymbol{y})$$ has an unbiased estimation as:

$$
\begin{array}{l}
\hat{\mathrm{I}}_{\mathrm{CLUB}}=\frac{1}{N} \sum_{i=1}^{N} \log p\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)-\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N} \log p\left(\boldsymbol{y}_{j} \mid \boldsymbol{x}_{i}\right) \\
=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N}\left[\log p\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)-\log p\left(\boldsymbol{y}_{j} \mid \boldsymbol{x}_{i}\right)\right] .
\end{array}
$$

Look at the form of the bound, it is exactly a log-ratio of the positive sample and the negative sample.

## 3.2. When $$p(y\mid x)$$ is Unknown

For some tasks, like if it is not a stochastic representation, then we have no access to the conditional distribution $$p\left(\boldsymbol{y} \mid \boldsymbol{x}\right)$$. Therefore, it is naturally to use a variational distribution $$q_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})$$ with parameter $$\theta$$. Therefore, the variational CLUB is:

$$
\mathrm{I}_{\mathrm{vCLUB}}(\boldsymbol{x} ; \boldsymbol{y}):= \mathbb{E}_{p(\boldsymbol{x}, \boldsymbol{y})}\left[\log q_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right] -\mathbb{E}_{p(\boldsymbol{x})} \mathbb{E}_{p(\boldsymbol{y})}\left[\log q_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right].
$$

Similarly, when we are using multiple samples, the bound becomes:

$$
\begin{aligned}
&\hat{\mathrm{I}}_{\mathrm{vCLUB}}=\frac{1}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N}\left[\log q_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)-\log q_{\theta}\left(\boldsymbol{y}_{j} \mid \boldsymbol{x}_{i}\right)\right]\\
&=\frac{1}{N} \sum_{i=1}^{N}\left[\log q_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)-\frac{1}{N} \sum_{j=1}^{N} \log q_{\theta}\left(\boldsymbol{y}_{j} \mid \boldsymbol{x}_{i}\right)\right]
\end{aligned}
$$

**WARNING**: No more guarantee on the bound. The proof of the power of this bound is based on the expressiveness of the neural network, which is not fully guaranteed as well.

In calculation, this sample-based method needs an $$\mathcal{O}\left(N^{2}\right)$$ because of the second term.

Authors propose another simpler bound, vCLUB-S (sampled vCLUB):

$$
\hat{\mathrm{I}}_{\mathrm{vCLUB}-\mathrm{S}}=\frac{1}{N} \sum_{i=1}^{N}\left[\log q_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}_{i}\right)-\log q_{\theta}\left(\boldsymbol{y}_{k_{i}^{\prime}} \mid \boldsymbol{x}_{i}\right)\right],
$$

which is unbiased with the computation complexity deduced to $$\mathcal{O}(N)$$. The negative pairs computation originally need all paired computation, which now only needs ONE pair. This property does not work for L1Out bound because in the denominator, there is a mean calculation within a $$\log$$, which if using single sample to estimate, is biased due to the Jensen's Inequality (again).

# 4. Experiment on Domain Adaption

Domain adaption tasks can be solved if the representation of an image can be disentangled in terms of the domain information and the semantic information.

Applying two encoders, a content encoder and a domain encoder, then we can apply the CLUB bound to minimize the mutual information between the content representation and the domain presentation. These two representation should be disentangled.
