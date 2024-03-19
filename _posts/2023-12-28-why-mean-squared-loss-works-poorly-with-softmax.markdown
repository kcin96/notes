---
layout: sidebar
title:  "Why  mean squared error loss works poorly with softmax."
date:   2023-12-29 21:16:00 +0100
categories: ML 
---

* toc
{:toc}

The softmax function is a popular function in the neural networks literature. To train networks with a softmax layer, it is almost always trained with cross entropy as the loss function. Networks with the softmax layer present is almost never seen to be trained using the mean squared error as the loss function. This section explores why that is the case by explaining why mean squared error loss works poorly with the softmax function.

## Softmax function

The definition of softmax function is as below.

$$\begin{aligned}
softmax \ function, \sigma{(x_i)}=\frac{e^{x_i}}{\sum_{i=1}^n {e^{x_i}}} \\
for \ i=1,2,...n
\end{aligned}$$

To do backpropagation, we will need to compute the gradients. The gradient for the softmax function can be derived using the quotient rule. For the proof, please refer to Eli's post [https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/). I will just quote the results here:

$$\begin{aligned}
\frac{\partial{\sigma(x_i)}}{\partial{x_j}}=\sigma(x_i)(\delta_{ij}-\sigma(x_j))=\begin{cases}
\sigma(x_i)(1-\sigma(x_j)), \ i=j\\
-\sigma(x_i)\sigma(x_j), \ i\neq j\\
\end{cases}
\end{aligned}$$

## Mean squared error loss + softmax
Let us assume that we have an arbitrary network with a few layers and a final softmax layer. We then decide to use the mean square error loss to train the network.
We can write down the loss function as
$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{\sigma(x)} - \boldsymbol{y}||^2_2 \\
\end{aligned}$$
where $\boldsymbol{y}$ is the target vector. For simplicity sake let us assume that $\boldsymbol{y}$ and $\boldsymbol{\sigma(x)}$ are 2x1 vectors (n=2).
To do backpropagation, we take the partial derivative of L with respect to $\boldsymbol{\sigma(x)}$.
$$\begin{aligned}
\frac{\partial{L}}{\partial{\boldsymbol{\sigma(x)}}}&=(\boldsymbol{\sigma(x)}-\boldsymbol{y})^T\\
&=\begin{bmatrix}
\frac{\partial{L}}{\partial{\sigma(x_1)}} \ \frac{\partial{L}}{\partial{\sigma(x_2)}}...\frac{\partial{L}}{\partial{\sigma(x_n)}}
\end{bmatrix}\\
&=\begin{bmatrix}
\sigma(x_1)-y_1 \ \ \ \sigma(x_2)-y_2 \ \ ...\ \sigma(x_n)-y_n 
\end{bmatrix}\\
\end{aligned}$$
Then to get $\frac{\partial{L}}{\partial{x_j}}$
$$\begin{aligned}
\frac{\partial{L}}{\partial{x_j}}&=
\frac{\partial{L}}{\partial{\boldsymbol{\sigma(x)}}}\frac{\partial{\boldsymbol{\sigma(x)}}}{\partial{x_j}}\\
\frac{\partial{\boldsymbol{\sigma(x)}}}{\partial{x_j}}&=
\begin{bmatrix}
\frac{\partial{\sigma(x_1)}}{\partial{x_j}}  \\ \frac{\partial{\sigma(x_2)}}{\partial{x_j}} \\.\\.\\.\\ \frac{\partial{\sigma(x_n)}}{\partial{x_j}}
\end{bmatrix}\\
\frac{\partial{L}}{\partial{x_1}}&=
\frac{\partial{L}}{\partial{\sigma(x_1)}}\frac{\partial{\sigma(x_1)}}{\partial{x_1}}+\frac{\partial{L}}{\partial{\sigma(x_2)}}\frac{\partial{\sigma(x_2)}}{\partial{x_1}}\\
&=(\sigma(x_1)-y_1)(\sigma(x_1)(1-\sigma(x_1)))+(\sigma(x_2)-y_2)(-\sigma(x_1)\sigma(x_2))\\
\frac{\partial{L}}{\partial{x_2}}&=\frac{\partial{L}}{\partial{\sigma(x_1)}}\frac{\partial{\sigma(x_1)}}{\partial{x_2}}+\frac{\partial{L}}{\partial{\sigma(x_2)}}\frac{\partial{\sigma(x_2)}}{\partial{x_2}}\\
&=(\sigma(x_1)-y_1)(-\sigma(x_1)\sigma(x_2))+(\sigma(x_2)-y_2)(\sigma(x_2)(1-\sigma(x_2)))\\
\end{aligned}$$
Now due to the exponential term in the softmax function, if one of the $x_i$ is significantly larger than the other $x_i$s it will completely dominate over the others i.e. the softmax of that big $x_i \approx 1$  and all other terms $\approx 0$ (as the sum all softmax terms = 1). For example,

$$\begin{aligned}
x_1=0.2,x_2=5,x_3=0.3\\
softmax(x_1)&=0.00809\\
softmax(x_2)&=0.983\\
softmax(x_3)&=0.00894\\

\end{aligned}$$

Julia code computation:
{% highlight julia %}
function softmax(x)
    x = x .- maximum(x)
    return exp.(x) ./ sum(exp.(x))
end

softmax([0.2,5,0.3])
{% endhighlight %}


Looking the $\frac{\partial{L}}{\partial{x_1}}$,$\frac{\partial{L}}{\partial{x_2}}$ equations above,
$$\begin{aligned}
if \ \sigma(x_1)\approx 0, \frac{\partial{L}}{\partial{x_1}} \approx 0\\
if \ \sigma(x_1)\approx 1 \implies \sigma(x_2) \approx 0, \frac{\partial{L}}{\partial{x_1}} \approx 0\\
if \ \sigma(x_2)\approx 0, \frac{\partial{L}}{\partial{x_2}} \approx 0\\
if \ \sigma(x_2)\approx 1 \implies \sigma(x_1) \approx 0, \frac{\partial{L}}{\partial{x_2}} \approx 0\\
\end{aligned}$$

In both cases, the gradient vanishes and no matter what layers we have before the softmax layer, it doesn't matter as the gradient being passed on is 0. So we cannot backpropagate the MSE errors back through the softmax layer when the softmax outputs saturates at 0 or 1. Hence, training can be difficult with these cases as the network gets stuck and stops learning. 

## Cross Entropy loss + softmax
Now let us go through the same process as above but with the cross entropy loss. Cross entropy function is defined as
$$\begin{aligned}
L = -\sum_{i=1}^n {p_i}lnq_i \\
\end{aligned}$$
Substituting $p_i$ with $y_i$ and $q_i$ by $\sigma(x)$ 
$$\begin{aligned}
L &= -\sum_{i=1}^n {y_i}ln\sigma(x_i) \\
\frac{\partial{L}}{\partial{\boldsymbol{\sigma(x)}}}&=
\begin{bmatrix}
\frac{\partial{L}}{\partial{\sigma(x_1)}} \ \frac{\partial{L}}{\partial{\sigma(x_2)}}...\frac{\partial{L}}{\partial{\sigma(x_n)}}
\end{bmatrix}\\
&=\begin{bmatrix}
\frac{-y_1}{\sigma(x_1)} \ \ \frac{-y_2}{\sigma(x_2)} \ ...\ \frac{-y_n}{\sigma(x_n)}
\end{bmatrix}\\
\end{aligned}$$
Then to get $\frac{\partial{L}}{\partial{x_j}}$
$$\begin{aligned}
\frac{\partial{L}}{\partial{x_j}}&=
\frac{\partial{L}}{\partial{\boldsymbol{\sigma(x)}}}\frac{\partial{\boldsymbol{\sigma(x)}}}{\partial{x_j}}\\
\frac{\partial{\boldsymbol{\sigma(x)}}}{\partial{x_j}}&=
\begin{bmatrix}
\frac{\partial{\sigma(x_1)}}{\partial{x_j}}  \\ \frac{\partial{\sigma(x_2)}}{\partial{x_j}} \\.\\.\\.\\ \frac{\partial{\sigma(x_n)}}{\partial{x_j}}
\end{bmatrix}\\
\frac{\partial{L}}{\partial{x_1}}&=
\frac{\partial{L}}{\partial{\sigma(x_1)}}\frac{\partial{\sigma(x_1)}}{\partial{x_1}}+\frac{\partial{L}}{\partial{\sigma(x_2)}}\frac{\partial{\sigma(x_2)}}{\partial{x_1}}\\
&=(\frac{-y_1}{\sigma(x_1)})(\sigma(x_1)(1-\sigma(x_1)))+(\frac{-y_2}{\sigma(x_2)})(-\sigma(x_1)\sigma(x_2))\\
&=-y_1+y_1(\sigma(x_1))+y_2(\sigma(x_1))\\
&=\sigma(x_1)-y_1\\
\frac{\partial{L}}{\partial{x_2}}&=\frac{\partial{L}}{\partial{\sigma(x_1)}}\frac{\partial{\sigma(x_1)}}{\partial{x_2}}+\frac{\partial{L}}{\partial{\sigma(x_2)}}\frac{\partial{\sigma(x_2)}}{\partial{x_2}}\\
&=(\frac{-y_1}{\sigma(x_1)})(-\sigma(x_1)\sigma(x_2))+(\frac{-y_2}{\sigma(x_2)})(\sigma(x_2)(1-\sigma(x_2)))\\
&=y_1(\sigma(x_2))+y_2(\sigma(x_2))-y_2\\
&=\sigma(x_2)-y_2\\
\end{aligned}$$

Looking the $\frac{\partial{L}}{\partial{x_1}}$,$\frac{\partial{L}}{\partial{x_2}}$ equations above,
$$\begin{aligned}
if \ \sigma(x_1)\approx 0, \frac{\partial{L}}{\partial{x_1}} \approx -y_1\\
if \ \sigma(x_1)\approx 1, \frac{\partial{L}}{\partial{x_1}} \approx 1-y_1\\
if \ \sigma(x_2)\approx 0, \frac{\partial{L}}{\partial{x_2}} \approx -y_2\\
if \ \sigma(x_2)\approx 1, \frac{\partial{L}}{\partial{x_2}} \approx 1-y_2\\
\end{aligned}$$

Unlike in the mean squared error case, we can see that the gradients $\frac{\partial{L}}{\partial{x_j}}$ do not disappear when the softmax values are saturated at 0 or 1. This is due to the fact that the derivative of the natural logarithm results in a denominator term that cancels out the softmax term in the $\frac{\partial{\sigma(x_i)}}{\partial{x_j}}$. 

## Summary
In summary, mean square error do not work well with the softmax function due to vanishing gradients as the softmax output saturates. This can also be extended to other loss functions such as the L1-norm where the derivatives of these functions do not cancel out the effect of the softmax (product of gradient terms). In contrast, with the cross entropy function, it has the property where it kind of 'cancel' out the softmax saturation values due to the presence of the denominator term in the gradient. 

## References
1. [http://www.deeplearningbook.org](http://www.deeplearningbook.org)
2. [https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

