---
layout: post
title:  "Understanding Regularization"
date:   2023-08-22 21:16:00 +0100
categories: 
---

## Least Squares
Let say we are doing a physics experiment to determine the gravitational acceleration $g$ by the falling ball experiment. Now from physics we know that the equation of motion that governs this experiment is $$\begin{equation}s=ut+\frac{1}{2}gt^2 \tag{1}\label{1}\end{equation}$$ In an ideal world, when we drop the ball from a particular height $s$ and measure the time $t$ it takes to travel $s$, we will find that the $t$ s fit the equation. In practice however, it could be that due to errors in the experiment (random/systematic/human reaction time), the data we obtain may not exactly conform to (\ref{1}). Nonetheless, it is possible that we would like to use our measured data and fit a line through them (build a model).



{% highlight ruby %}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

np.random.seed(10)
t = np.linspace(0,3,10)
g = -9.81
u = 0

s = u*t + 0.5*g*t**2
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(t,s,label="$s=ut+1/2 gt^2$")
ax.plot(t,s+np.random.randn(len(t)),'x',label="Measured data")
ax.set_xlabel("Time t/s")
ax.set_ylabel("Displacement s/m")
ax.legend();

{% endhighlight %}
![Image 1](/assets/images/regularization-1.png)

For our model, let $s_t$ be the true displacement data points that fits (1), $y_t$ the displacement results from our experiment (measured data). $y_t$ can be written as a linear combination of powers of time $y_t=w_0+w_1t+w_2t^2+...$, where $w_n$ are weights for each $t^n$ for $n=0,1,2...N$. If we define $\phi(t)$ as a vector function of $t$ s, $y_t$ can be written vectorially as $\boldsymbol{w}^T\phi(t)$. With our experiment, the weights are the parameters to be determined from our measured data. A good model will have $w_o\approx 0$, $w_1\approx 0$, $w_2\approx 0.5*9.81$ and all other weights = $0$. 

To determine the error $E_D$ between true data (target) and measured data, a common metric is to use the sum-of-squares error (SSE). $SSE=\sum_{t=1}^{N}(s_t-y_t)^2$ So when the all $y_t$ equals the $s_t$, i.e. our measurements match the true data, we get a SSE of zero. 

As a simple example, let $N=1, t=1, s_1=7, w_0=0, w_1\in\{-100...100\},w_2\in\{-100...100\}$. The code below plots the sum-of-squares error for all $w_1,w_2$ pairs between -100 and 100.

{% highlight ruby %}
from mpl_toolkits.mplot3d import axes3d

# for simplicity, assume N=1, t=1, s_1=7, w_0=0
s_1 = 7
t = 1
w_1 = np.linspace(-100,100,200)
w_2 = np.linspace(-100,100,200)

[W_1, W_2] = np.meshgrid(w_1, w_2)
f = (s_1 - (W_1*t+W_2*t**2))**2 

ax = plt.figure(figsize=(4,4)).add_subplot(projection='3d')
ax.plot_surface(W_1,W_2,f)
ax.set_xlabel("$w_1$")
ax.set_ylabel("$w_2$")
ax.set_zlabel("$SSE$")
ax.contour(W_1,W_2,f,zdir='z',offset=0)
ax.view_init(elev=15, azim=20)


{% endhighlight %}
![Image 2](/assets/images/regularization-2.png)

The contour lines for various SSE values (level sets) are plotted as coloured lines on the $w_1w_2$ plane.

More generally, the SSE between target and data can be written as $E_D(\boldsymbol{w})=\frac{1}{2}\sum_{n=1}^{N}(t_n-y_n)^2$ where $N$: Number of data points, $t_n$: target data, $y_n=\boldsymbol{w}^T\phi(\boldsymbol{x}_n)$: A linear combination of basis functions $\phi(x)$.

With the $E_D$ (loss function) defined, we may proceed to build a regression model. Quite often we find that with limited data, our model overfits. This is where regularization comes in. Regularization is a technique to combat overfitting.

## Regularized Least Squares 
### Regularized Error function

$$E_R(\boldsymbol{w})=\frac{1}{2}\sum_{n=1}^{N}(t_n-\boldsymbol{w}^T\phi(\boldsymbol{x}_n))^2+\lambda \boldsymbol{w}^T\boldsymbol{w} \tag{2}\label{2}$$

The regularized error function consists of 2 terms - the error between target and data and the squared weights (L2 penalty function).

This is related to solving the constrained optimization problem

$$\begin{equation}minimize\ \frac{1}{2}\sum_{n=1}^{N}(t_n-\boldsymbol{w}^T\phi(\boldsymbol{x}_n))^2 \\
subject\; to\; \boldsymbol{w}^T\boldsymbol{w}\le\eta \end{equation}$$
where $\eta$ is a parameter. We can write the above problem in terms of the Lagrange function $L(\boldsymbol{w},\lambda') = f(\boldsymbol{w}) + \lambda'g(\boldsymbol{w})$ 
where $$\begin{equation}f(\boldsymbol{w}) = \frac{1}{2}\sum_{n=1}^{N}(t_n-\boldsymbol{w}^T\phi(\boldsymbol{x}_n))^2\end{equation}$$, $g(\boldsymbol{w})=\boldsymbol{w}^T\boldsymbol{w}-\eta $. 
Minimizing $$\begin{equation}L(\boldsymbol{w},\lambda' ) = \frac{1}{2}\sum_{n=1}^{N}(t_n-\boldsymbol{w}^T\phi(\boldsymbol{x}_n))^2  + \lambda'(\boldsymbol{w}^T\boldsymbol{w}-\eta) \tag{3}\end{equation}$$ is equivalent to minimizing (\ref{2}) if $\lambda'=\lambda$. To see this, when $\lambda'=\lambda$, notice that the lagrangian has an extra $-\lambda\eta$ constant term. The $\boldsymbol{w} $ that minimizes the lagrangian, $\boldsymbol{w}^\*$ is the same solution as minimizing (\ref{2}) albeit with a $L(\boldsymbol{w}^\*)$ value $\lambda\eta$ less than $E_R(\boldsymbol{w^\*})$.

Let $\boldsymbol{w}$ be a weight vector consisting of 2 variables, $w_1, w_2$. Hence, $$\begin{equation}\boldsymbol{w}^T\boldsymbol{w}=\begin{bmatrix}w_1 & w_2\end{bmatrix}
\times
\begin{bmatrix}w_1 \\ w_2\end{bmatrix}=w_1^2+w_2^2 \end{equation}$$.
The figure below shows a plot of $w_1$ vs $w_2$ with $\eta=1$ and $\eta=4$. 

{% highlight ruby %}
#set weight order
q = 2

#plotting constraint with eta=1
eta = 1
fig, ax = plt.subplots(figsize=(4, 4))
w1 = np.linspace(-1,1,201)
w2 = np.power(eta - np.power(np.abs(w1),q), 1/q)
ax.plot(w1,w2,color='b')
ax.plot(w1,-w2,color='b')
ax.annotate(xy=(0.7,-1),text=r'$\eta=1$')

#plotting constraint with eta=4
eta = 4
w1 = np.linspace(-2,2,201)
w2 = np.power(eta - np.power(np.abs(w1),q), 1/q)
ax.plot(w1,w2,color='b')
ax.plot(w1,-w2,color='b')
ax.annotate(xy=(1.5,-1.5),text=r'$\eta=4$')
ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$');
ax.grid()
{% endhighlight %}

![Image 3](/assets/images/regularization-3.png)

The internal area of each circle represents the set of $w_1,w_2$ that satisfies the constraint (inequality) $\boldsymbol{w}^T\boldsymbol{w}\le\eta$. 

To understand how the constraint function affects the weight parameters, let us illustrate with an example. Let say our target line is $y=6x + x^2$. 3 true $x$, $x^2$ points are $[4,16],[1,1],[7,49]$. And let say our measured data are $[2,7],[1,5],[6,10]$. The code below illustrates our example.

{% highlight ruby %}
# target line y=6x+x^2
target = np.array([[4,16],[1,1],[7,49]])
x = np.linspace(-10,20,100)
y = np.linspace(-10,20,100)

[W1, W2] = np.meshgrid(x, y)
d = np.array([[2,7],[1,5],[6,10]]).T

m = np.matmul(np.concatenate((W1.reshape(-1,1),W2.reshape(-1,1)),axis=1),d)
diff = (np.matmul(np.array([6,1]),target.T)-m)
f = 0.5*(np.sum(np.square(diff),axis=1).reshape(W1.shape[0],-1))

# Plotting 3D SSE 
ax = plt.figure(figsize=(4,4)).add_subplot(projection='3d')
ax.plot_surface(W1,W2,f)
ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$')
ax.set_zlabel(r'$SSE$')
ax.contour(W1,W2,f,zdir='z',offset=0)
ax.view_init(elev=10, azim=-45)
plt.show()

#From graph, min located approx at (15,1)
fig, ax = plt.subplots(figsize=(4,4))
cp = ax.contour(W1,W2,f,levels=[100,500,1000,2000,3000])
ax.clabel(cp, cp.levels)

#plotting constraint curves for eta = 4,16,36
eta = 4
w1 = np.linspace(-2,2,201)
w2 = np.power(eta - np.power(np.abs(w1),q), 1/q)
ax.plot(w1,w2,color='b')
ax.plot(w1,-w2,color='b')
ax.annotate(xy=(-1.5,-1.3),xytext=(-8,-6),arrowprops=dict(facecolor='black', 
                                                     shrink=1, 
                                                     width=0.1,
                                                     headwidth=5),
                                                     text=r'$\eta=4$')

eta = 16
w1 = np.linspace(-4,4,201)
w2 = np.power(eta - np.power(np.abs(w1),q), 1/q)
ax.plot(w1,w2,color='b')
ax.plot(w1,-w2,color='b')
ax.annotate(xy=(-1,-4),xytext=(-8,-8),arrowprops=dict(facecolor='black', 
                                                     shrink=1, 
                                                     width=0.1,
                                                     headwidth=5),
                                                     text=r'$\eta=16$')

eta = 36
w1 = np.linspace(-6,6,201)
w2 = np.power(eta - np.power(np.abs(w1),q), 1/q)
ax.plot(w1,w2,color='b')
ax.plot(w1,-w2,color='b')
ax.annotate(xy=(0,-6),xytext=(-8,-10),arrowprops=dict(facecolor='black', 
                                                     shrink=1, 
                                                     width=0.1,
                                                     headwidth=5),
                                                     text=r'$\eta=36$')

ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$')
plt.show()

{% endhighlight %}

![Image 4](/assets/images/regularization-4.png)\

The blue surface shows the unregularized error function. 

![Image 5](/assets/images/regularization-5.png)

Next, we superimpose the constraint curves on the unregularized error function contour plot as shown in the figure above. The coloured contour lines are the level sets for the unregularized error function while the dark blue circular plots are the constraint curves for $\eta = 4,16,36$. From the contour plot, we observe that
1. The minimum point for the unregularized error function at approximately at (15,1).
2. With $\eta=36$, the constraint curve intersects the contour at about 500. i.e. the constraint curve 'pulls' the minimum point towards the left of the $w_1w_2$ plane as only points within the circle are valid (points that satisfies $\boldsymbol{w}^T\boldsymbol{w}\le 36$). 
3. With $\eta=4$, the constraint curve intersects the contour at about 3000. So as we decrease $\eta$, the loss value increases but the magnitude of $w_1$ and $w_2$ decreases: (2,2). 
  
From the above observation, we can see that when we set a constraint that the magnitude of $\boldsymbol{w}$ be small, the solution we find will tend to have a smaller magnitude than that of the unregularized error function. 

Now with these insights, let us return to (\ref{2}). While the lagrangian is similar to (\ref{2}), in machine learning context, the second term in (1) is usually referred to as the penalty function. The regularization coefficient $\lambda$ is usually a hyperparameter set by experimentation. This $\lambda$ factor is something like a weight. For $\lambda = 0$, the is no regularization applied. For a small $\lambda$ the penalty function has a small overall effect on the overall loss function, $E_R(\boldsymbol{w})$. For a large $\lambda$ we are enforcing that the penalty function is an important factor to minimize i.e. the weights should be small. Due to this property, it is also known as weight decay in machine learning literature. Regularization reduces overfitting as it encourages smaller coefficients. For instance, fitting a straight line $w_0+w_1x+w_2x^2+...$, we would expect that the weights of the higher order terms to be close to zero. If the model overfits, we would have non-zero coefficients for the higher powers. e.g. instead of drawing a straight line, we can have a quadratic/cubic curve passing through all points. 

### Example: Regularization in action 
Let us say that we are trying to fit a quadratic curve $f(x)=7+x+3x^2$. For simplicity, we are given only 6 $x$ points:$ -3, -1, 2, 3, 5, 7$. Let us model the data using a 4th order polynomial - $w_0+w_1x+w_1x^2+w_3x^3+w_4x^4 = \boldsymbol{w}^T\phi(x)$, where $\phi(x)=\begin{bmatrix}1 \\ x \\ x^2 \\ x^3 \\ x^4\end{bmatrix}$,  $\boldsymbol{w}=\begin{bmatrix}w_0 \\ w_1 \\ w_2 \\ w_3 \\ w_4\end{bmatrix}$. Let us set the regularization coefficient $\lambda = 5$ and initialize $\boldsymbol{w}=\begin{bmatrix}10 \\ 10 \\ 10 \\ 10 \\ 9\end{bmatrix}$. With these values, we can compute the loss function (\ref{2}).

{% highlight ruby %}
x = np.array([-3,-1,2,3,5,7])  #initialize x 
w = np.array([10,10,10,10,9])  #initialize weight
lamb = 5 #regularization coefficient
{% endhighlight %}

To find the $\boldsymbol{w}$ that minimizes the loss function (\ref{2}), we use stochastic gradient descent (SGD). 
Gradient of (2):
$\nabla{E_R(\boldsymbol{w})}=-(f(x)-\boldsymbol{w}^T\phi(x))\phi(x)+2\lambda \boldsymbol{w}$

The update step:
$\boldsymbol{w^{i+1}}=\boldsymbol{w^i}-\frac{\alpha}{N} \nabla{E_R(\boldsymbol{w})}$

Learning rate: $\alpha = 10^{-6}$ (Note: Hyperparameter set by experimenting various values.)

Number of samples: $N$

Number of iterations: $100,000$

{% highlight ruby %}
# SGD to obtain weights
alpha = 0.000001 #learning rate
for iter in range(100000):
    L = 0 #initialize loss
    for x_i in x:
        phi_x = np.array([1, x_i, x_i**2, x_i**3,x_i**4])
        weighted_s = np.matmul(w,phi_x)
        L += 0.5*(7+x_i+3*x_i**2-weighted_s)**2 + lamb*np.sum(w.T*w) #Loss function
        dL = -phi_x*(7+x_i+3*x_i**2-weighted_s) + lamb*2*w #Gradient of loss
        w = w-alpha*(dL)/len(x)  #Update step
    
    print("Loss:",L)
    print (w)
{% endhighlight %}


Putting it all together
{% highlight ruby %}
# Example of regularization vs no regularization
x = np.array([-3,-1,2,3,5,7])  #initialize x 
w = np.array([10,10,10,10,9])  #initialize weights
alpha = 0.000001 #learning rate
lamb = 5 #regularization coefficient

# SGD to obtain weights
for iter in range(100000):
    L = 0 #initialize loss
    for x_i in x:
        phi_x = np.array([1, x_i, x_i**2, x_i**3,x_i**4])
        weighted_s = np.matmul(w,phi_x)
        L += 0.5*(7+x_i+3*x_i**2-weighted_s)**2 + lamb*np.sum(w.T*w) #Loss function
        dL = -phi_x*(7+x_i+3*x_i**2-weighted_s) + lamb*2*w #Gradient of loss
        w = w-alpha*(dL)/len(x)  #Update step
    
    print("Loss:",L)
    print (w)


#Results
#no regularization: lamb = 0 
#after 100,000 iterations
#Loss: 113.81919356847847
#[ 9.14084877  8.18992779  2.46073974 -0.65050656  0.08209003]


#with regularization: lamb = 5 
#after 100,000 iterations
#Loss: 954.2707341632134
#[ 3.58727023  3.14255901  2.5549417  -0.1588695   0.02697015]
{% endhighlight %}


Results:

After 100,000 iterations

|  | Loss | w |
| ---- | --- | --- |
| No regularization: $\lambda = 0$ |  113.8 | [ 9.14084877  8.18992779  2.46073974 -0.65050656  0.08209003] |
| With regularization: $\lambda = 5$ | 954.3 | [ 3.58727023  3.14255901  2.5549417  -0.1588695   0.02697015] |

We can see from the above table that regularization results in smaller coefficients. The penalty function penalizes large weight values and results in smaller coefficents. We can see that the 3rd and 4th order coefficients are driven close to 0 in the regularized case. 

Let us plot the graphs for the actual line, unregularized and regularized case.

{% highlight ruby %}
#Plotting $f=7+x+3x^2$, with lambda=0, lambda=5 
x = np.linspace(-5,5,50)
fig, ax = plt.subplots(figsize=(4, 4))
f = 7+x+3*x**2
ax.plot(x,f,label="$f=7+x+3x^2$")
ax.plot(x, 9.14084877+8.18992779*x+2.46073974*x**2-0.65050656*x**3+0.08209003*x**4,label="$\lambda=0$ (no regularization)")
ax.plot(x, 3.58727023+3.14255901*x+2.5549417*x**2-0.1588695*x**3+0.02697015*x**4,label="$\lambda=5$ (with regularization)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend();
{% endhighlight %}

![Image 6](/assets/images/regularization-6.png)

From the above plot, the green line ($\lambda=5$) is closer to the actual line $f$ (blue) than the orange line ($\lambda=0$) which we are trying to learn. 

## L1 regularization
The other penalty function commonly used is the L1 norm. So instead of penalizing the squared weights (L2 norm), we penalize the L1 norm of the weights. So instead of (\ref{2}), we have

$$E_R(\boldsymbol{w})=\frac{1}{2}\sum_{n=1}^{N}(t_n-\boldsymbol{w}^T\phi(\boldsymbol{x}_n))^2+\lambda |\boldsymbol{w}| \tag{4}$$

The L1 regularizer tend to produce sparse solutions i.e. more coefficients are driven to zero. To see this, compare the quadratic function and absolute valued function.

{% highlight ruby %}
# Plotting the quadratic function and absolute valued function
w = np.linspace(-2,2,51)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(w,np.absolute(w),label="$f=|w|$")
ax.plot(w, np.square(w),label="$f=w^2$")
ax.set_xlabel("w")
ax.set_ylabel("f(w)")
ax.grid()
ax.legend();
{% endhighlight %}

![Image 7](/assets/images/regularization-7.png)

Looking at the graph, in the range $-1 < w < 1$, for the L1 norm, the gradient is still a constant, whereas with the L2 norm, the gradient decreases with every step moving closer to zero. So when we do gradient descent, L1 will take a constant step forward in every iteration towards 0. With L2 norm, as the gradient decreases as we move to 0, each step size gets smaller and smaller with each iteration. The following table 
compares the $w$ for 10 iterations of gradient descent for $f_{L1}(w)=|w|$, $f_{L2}(w)=w^2$ with step $s = 0.1$. 

| Iteration | $\nabla f(w)=1$,  $w^{i+1} = w^{i} - s$   | $\nabla f(w)= 2w$, $w^{i+1} = w^{i} - 2sw^{i}$  |
| --- | --- | ---|
| 0 | $w = 1$ | $w=1$ |
| 1 | $w = 0.9$ | $w = 0.8$ |
| 2 | $w = 0.8$ | $w = 0.8-0.2(0.8)=0.64$ |
| 3 | $w = 0.7$ | $w = 0.64-0.2(0.64)=0.512$ |
| | ... | ... |
|10|$w = 0$ | $w = 0.1073741824$ |

{% highlight ruby %}
#Computing the w values for L1 & L2 
def recursel1(w,s):
    return w - s

def recursel2(w,s):
    return w - s*w

l_1 = [1]  #initialize w = 1 at iteration 0
l_2 = [1]  #initialize w = 1 at iteration 0

#L1 iteration
w = 1
for iter in range(10):
    w = recursel1(w, 0.1)
    l_1.append(w)

#L2 iteration
w = 1
for iter in range(10):
    w = recursel2(w, 0.2)
    l_2.append(w)

# Plotting the w values for L1 & L2 for 10 iterations
w = np.linspace(0,1.5,51)
x = np.linspace(0,10,11)
fig, ax = plt.subplots(figsize=(4, 4))

ax.bar(x,l_1,width=0.3,label="$w$ for L1")
ax.bar(x,l_2,width=0.15,label="$w$ for L2")
ax.set_xlabel("Iteration")
ax.set_ylabel("w")
ax.set_xticks(np.arange(0,11))
ax.set_yticks(np.arange(0,1.1,0.1))
ax.grid(True)
ax.legend();

{% endhighlight %}

![Image 8](/assets/images/regularization-8.png)

As the graph above indicates, the L1 norm has $w$ driven to 0 at iteration 10, whereas with L2, the closer we get to 0, the slower the decrease in $w$ as the step reduction gets smaller with each iteration.

## References 
1. Christopher M. Bishop. Pattern Recognition and. Machine Learning
2. [https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec06-slides.pdf](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec06-slides.pdf) 