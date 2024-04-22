---
layout: sidebar
title:  "Conjugate Gradients"
date:   2024-04-08 21:16:00 +0100
categories: ML 
---

* toc
{:toc}
<script src="{{ "assets/js/sim.js" | relative_url }}" type="text/javascript"></script>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
<script src="https://unpkg.com/mathjs/lib/browser/math.js"></script>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>



## A-conjugacy
We begin with a unit circle plot as shown below. We draw 2 perpendicular (orthogonal) vectors (orange) on the circumference. We can verify that these vectors are orthogonal by taking the dot product and obtaining 0. 

{%highlight julia%}
using Plots

# Get circle points
x = collect(-1:0.01:1)
y = sqrt.(1 .- x.^2)
x = append!(x, -x)
y = append!(y, -y)
z = [x y]'

# Plot circle and perpendicular arrow vectors
plot(z[1,:], z[2,:], aspect_ratio=:equal)
v = [-1 -1 1 1 0 0 0 0;0 0 0 0 1 1 -1 -1]
Δv = [0.25 0.25 -0.25 -0.25 0.25 -0.25 0.25 -0.25;-0.25 0.25 0.25 -0.25 -0.25 -0.25 0.25 0.25]
quiver!(v[1,:], v[2,:], quiver=(Δv[1,:], Δv[2,:]))

{%endhighlight%}

![Image 1]({{site.baseurl}}/assets/images/conjugate-gradients/circ1.svg)

Next, let us stretch the space along the x-axis by a factor of $2$. This stretching operation can be thought of as warping our Cartesian plane. We see this 'stretch' if we continue to use our x & y axis in terms of the same units. e.g. same units of meters on x & y axis. 

Now, how to 'stretch' the x-axis by 2? This can be done by a transformation matrix $T$. 

$$\begin{aligned}
\boldsymbol{T} &=
\begin{bmatrix}
2 \ 0 \\ 0 \ 1 
\end{bmatrix}\\
\boldsymbol{T}\boldsymbol{z} &=
\begin{bmatrix}
2 \ 0 \\ 0 \ 1 
\end{bmatrix}
\begin{bmatrix}
x \\ y 
\end{bmatrix}
\end{aligned}$$

{%highlight julia%}
# Transformation matrix
T = [2 0;0 1]
tz = T*z
v_trans = T*v
Δv_trans = T*Δv

# Plot scaled circled and resulting arrow vectors
plot(tz[1,:], tz[2,:], aspect_ratio=:equal)
quiver!(v_trans[1,:], v_trans[2,:], quiver=(Δv_trans[1,:], Δv_trans[2,:]))
{%endhighlight%}

![Image 2]({{site.baseurl}}/assets/images/conjugate-gradients/circ2.svg)

However, if we think of the transformed plane as having the x axis scaled by 2, and keeping y the same unit, we will still have a circle on a transformed plane. This is similar idea to log-linear graphs, or a graph of $cm$ vs $mm$.

{%highlight julia%}
# scaled x-axis by 2
plot(tz[1,:], tz[2,:], aspect_ratio=:2)
{%endhighlight%}

![Image 3]({{site.baseurl}}/assets/images/conjugate-gradients/circ3.svg)

The arrow vectors are $\boldsymbol{A}$ orthogonal vectors in the scaled space, as there exist an $\boldsymbol{A}$ matrix (inverse operation of $\boldsymbol{T}$) that reverses the transformation step ("unwarps our space"). And we know that in the unwarped space, the arrow vectors are orthogonal.

{%highlight julia%}
#= 
Vectors are 'A' orthogonal, as shown by taking the dot product 
of 'unwarped' vectors 
=#
(inv(T)*Δv_trans)[:,1]'*(inv(T)*Δv_trans)[:,2]

# Result:
# 0.0
{%endhighlight%}


### Interactive Demo
<div class="card" style="height:500px; width: 900px;" id="simbox">
    <div class="container text-center" style="padding:10px">
        <div class="row">
            <div class="col-3">
                <div class="form-control form-control-sm" style="height:450px; width: 200px;" >
                    <label for="customRangex" class="form-label">x range</label>
                    <p id="xscaleval">1</p>
                    <input type="range" class="form-range" min="-10" max="10" id="xscale" value=1>
                    <label for="customRangey" class="form-label">y range</label>
                    <p id="yscaleval">1</p>
                    <input type="range" class="form-range" min="-10" max="10" id="yscale" value=1>
                    <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" role="switch" id="autoscale_graph">
                    <label class="form-check-label">Autoscale graph</label>
</div>
                </div>
            </div>
            <div class="col-6" id="graphplot">
            </div>
        </div>
    </div>

</div>



Consider the quadratic problem\\
$f(\boldsymbol{x})=\frac{1}{2}\boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}^T\boldsymbol{x}+c$\\
where $\boldsymbol{A}$ is a symmetric positive definite matrix, $\boldsymbol{x}$ and $\boldsymbol{b}$ are vectors, c some arbitrary constant.
The gradient is \\
$$\begin{aligned}
\nabla \boldsymbol{f}_x &= \frac{1}{2}\boldsymbol{A}^T\boldsymbol{x}+\frac{1}{2}\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\\
\nabla \boldsymbol{f}_x &= \boldsymbol{A}\boldsymbol{x}-\boldsymbol{b} &(since \ \boldsymbol{A} \ is \ symmetric)\\
\boldsymbol{0} &=  \boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\\
\boldsymbol{A}\boldsymbol{x}&=\boldsymbol{b}\\
\end{aligned}$$
The global minimizer to the quadratic problem can be found by solving $\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ if $\boldsymbol{A}$ is positive definite ($\boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x}>0\ for\ all\ \boldsymbol{x}$)

Now for simplicity, I will abuse notation and treat unbold symbols as matrix/vectors.

Proof:\\
Let $x^*$ be the minimum point, $A$ positive definite matrix, $\delta $ be some perturbation.
$$\begin{aligned}
f(x^*+\delta) &= \frac{1}{2}(x^*+\delta)^TA(x^*+\delta)-b^T(x^*+\delta)+c\\
&= \frac{1}{2}(x^*)^TAx^*+\frac{1}{2}(x^*)^TA\delta+\frac{1}{2}\delta^TAx^*+\frac{1}{2}\delta^TA\delta-b^T(x^*+\delta)+c\\
&= \frac{1}{2}(x^*)^TAx^*-b^Tx^*+c+(\delta)^TAx^*-b^T\delta+\frac{1}{2}\delta^TA\delta\\
&= f(x^*)+\frac{1}{2}\delta^TA\delta\\
\end{aligned}$$

Since $\frac{1}{2}\delta^TA\delta>0$ due to positive definiteness, $x^*$ is the global minimum. 

Given $p = \\{d_1,d_2...d_n\\}$ of mutually conjugate vectors, $P$ forms a set of basis of $\mathbb{R}^n$ and $A$ to be symmetrical positive definite matrix. Positive definiteness implies strict convexity i.e. there is a minimum solution and no saddle points.

If $d_i^TAd_j=0$ then $d_i$ is $A$ conjugate to $d_j$.

Now since $P$ is a set of $n$ basis vectors, we can write 
$$\begin{aligned}
x^* = \sum_{i=1}^n\alpha_id_i
\end{aligned}$$
To solve for $\alpha_i$ we apply the A-conjugate property
$$\begin{aligned}
Ax^* &= \sum_{i=1}^n\alpha_iAd_i\\
d_j^TAx^* &= \sum_{i=1}^n\alpha_id_j^TAd_i\\
d_j^Tb &= \alpha_jd_j^TAd_j\\
\alpha_j &= \frac{d_j^Tb}{d_j^TAd_j}
\end{aligned}$$

## Orthogonal Directions
Orthogonal Direction is the idea that for each direction $d_i$ in a $\mathbb{R}^n$ space, the solution can be found in $n$ steps. This is done by having each iteration step through to the correct $x_i$ for the ith direction. e.g. If we imagine the direction vectors as the basis vectors, we are stepping through each basis vectors. So it takes 2 steps to get to the solution for a circular contour plot - 1st step to correct x coordinate, 2nd step to correct y coordinate.

## Conjugate Directions
Taking the idea of the circular contuor plot of orthogonal direction, we can apply that idea to the A-conjugate case.

Let residual = -gradient.
$$\begin{aligned}
x_{j+1} &= x_j+\alpha_jd_j\\
\alpha_j &= \frac{d^T_jr_j}{d_j^TAd_j}\\ 
where \ r_j &= b-Ax_j \ is \ the \ jth \ residual\\
\end{aligned}$$

Proof:
$$\begin{aligned}
x^*-x_0 &= \sum_{i=1}^n\alpha_i d_i\\
x_j-x_0 &= \sum_{i=1}^{j-1}\alpha_i d_i\\
d_j^TA(x^*-x_0) &= d_j^TA(\sum_{i=1}^n\alpha_i d_i)\\
d_j^TA(x^*-x_0) &= \sum_{i=1}^n\alpha_i d_j^T A d_i\\
d_j^TA(x^*-x_0) &= \alpha_j d_j^T A d_j\\
\alpha_j &= \frac{d_j^TA(x^*-x_0) }{d_j^T A d_j}\\
\alpha_j &= \frac{d_j^TA(x^*-x_j+x_j-x_0) }{d_j^T A d_j}\\
\alpha_j &= \frac{d_j^TA(x^*-x_j) }{d_j^T A d_j}\\  
\alpha_j &= \frac{d_j^T(b-Ax_j) }{d_j^T A d_j}\\  
\alpha_j &= \frac{d_j^Tr_j }{d_j^T A d_j}\\    
\end{aligned}$$

### Gram-Schmidt
Let ${u_0,u_2...u_{n-1}}$ be linearly independent vectors.

$$\begin{aligned}
d_i &= u_i+\sum_{k=0}^{i-1}\beta_{ik} d_k\\
d_j^TAd_i &= d_j^TAu_i+d_j^TA\sum_{k=0}^{i-1}\beta_{ik} d_k
\ where \ j<i\\
0 &= d_j^TAu_i+d_j^TA\sum_{k=0}^{i-1}\beta_{ik} d_k\\
0 &= d_j^TAu_i+d_j^TA\beta_{ij} d_j\\
\beta_{ij} &= -\frac{d_j^TAu_i}{d_j^TAd_j}\\
\beta_{ij} &= -\frac{u_i^TAd_j}{d_j^TAd_j}\\
\end{aligned}$$

## Conjugate Gradients
Conjugate gradients (CG) is essentially conjugate directions, setting $u$ to the residual $r$.

* Update x 
$$\begin{aligned}
x_{j+1} &= x_j+\alpha_jd_j\\
\alpha_j &= \frac{d_j^Tr_j }{d_j^T A d_j}\\    
\end{aligned}$$

* Update residual
$$\begin{aligned}
r_{j+1} &= -Ae_{j+1} \ (e_j=x_j-x^*)\\
r_{j+1} &= -A(e_j+\alpha_jd_j)\\
r_{j+1} &= r_j-\alpha_jAd_j\\
\end{aligned}$$

* Update direction vector.
As all $r_i$ are A orthogonal, $\beta_{ij} = -\frac{r_i^TAd_j}{d_j^TAd_j}$ becomes $\beta_{j+1} = -\frac{r_j^TAd_j}{d_j^TAd_j}$

$$\begin{aligned}
d_{j+1} &= r_{j+1}+\beta_{j+1} d_j\\
\beta_{j+1} &= -\frac{r_j^TAd_j}{d_j^TAd_j}\\
\end{aligned}$$

### Julia implementation
{%highlight julia%}
function conjugate_grad(A, x, b)
    r = -(A*x-b)
    d = r 
    while sum(abs2.(r))>0.1
        α = d'*r/(d'*A*d)
        println(x.+α*d)
        println(α)
        x = x .+ α*d
        r = r .- α*A*d

        β = -r'*A*d/(d'*A*d)
        d = r .+ β*d
        println(β)
        println("r:",r)
    end
    x
end

conjugate_grad([4 1;1 3.], [45,1.],[1,2.])
{%endhighlight%}

Output:
{%highlight julia%}
[4.261940357227161, -9.41083746426417]
0.2263225535709602
0.020816994193039673
r:[-6.636923964644495, 25.97057203556534]
[0.09090909090908994, 0.6363636363636367]
0.4016793265837188
-4.387386080429377e-17
r:[-1.865174681370263e-14, -3.552713678800501e-15]
2-element Vector{Float64}:
 0.09090909090908994
 0.6363636363636367 
{%endhighlight%}

### $\beta$ simplification
$\beta$ can be further simplified by the following steps.
$$\begin{equation}
r_{j+1} = r_j - \alpha_jAd_j \tag{1}\label{eq:residual}
\end{equation}$$

$$\begin{aligned}
r_i^Tr_{j+1} &= r_i^Tr_j - \alpha_jr_i^TAd_j\\
r_i^TAd_j &= \frac{r_i^Tr_j - r_i^Tr_{j+1} }{\alpha_i} \\
&= \frac{- r_i^Tr_{j+1} }{\alpha_i} \ for \  i=j+1\\
\end{aligned}$$

Using (\ref{eq:residual})  again
$$\begin{aligned}
r_{j+1} &= r_j - \alpha_jAd_j \\
d_j^Tr_{j+1} &= d_j^Tr_j - \alpha_jd_j^TAd_j\\
0 &= d_j^Tr_j - \alpha_jd_j^TAd_j\\
\alpha_jd_j^TAd_j &= d_j^Tr_j\\
using \ 
d_i^T &= u_i^T+\sum_{k=0}^{i-1}\beta_{ik} d_k\\
d_i^Tr_j &= u_i^Tr_j+r_j\sum_{k=0}^{i-1}\beta_{ik} d_k\\
d_i^Tr_j &= u_i^Tr_j\\
\alpha_jd_j^TAd_j &= r_j^Tr_j\\
\end{aligned}$$
Hence,
$$\begin{aligned}
\beta_{ij} &= -\frac{r_i^TAd_j}{d_j^TAd_j}\\
\beta_{j+1} &= \frac{r_{j+1}^Tr_{j+1}}{r_{j}^Tr_{j}}
\end{aligned}$$

### CG summary
To summarise the conjugate gradient algorithm,
<div class="card border-primary mb-3">
    <div class="card-header"><h5>Conjugate direction algorithm</h5></div>  
    <div class="card-body">
        Initialise direction vector to the residual
        $$\begin{aligned}d_0 = r_0
        \end{aligned}$$
        Update x 
        $$\begin{aligned}
        x_{j+1} &= x_j+\alpha_jd_j\\
        \alpha_j &= \frac{d_j^Tr_j }{d_j^T A d_j}\\    
        \end{aligned}$$
        Update residual
        $$\begin{aligned}
        r_{j+1} &= r_j-\alpha_jAd_j\\
        \end{aligned}$$
        Update direction vector.
        $$\begin{aligned}
        d_{j+1} &= r_{j+1}+\beta_{j+1} d_j\\
        \beta_{j+1} &= \frac{r_{j+1}^Tr_{j+1}}{r_{j}^Tr_{j}}
        \end{aligned}$$
    </div>
</div>

### Julia implementation example
{%highlight julia%}
function conjugate_grad2(A, x, b)
    r = -(A*x-b)
    d = r 
    while sum(abs2.(r))>0.1
        α = d'*r/(d'*A*d)
        println(x.+α*d)
        println(α)

        x = x .+ α*d
        r_new = r .- α*A*d

        β = r_new'*r_new/(r'*r)
        d = r_new .+ β*d

        r = r_new
        println(β)
        println("r:",r)
    end
    x
end

conjugate_grad2([4 1;1 3.], [-8,-8.], [1,2.])
{%endhighlight%}

Output:
{%highlight julia%}
[0.961248073959938, -0.5687211093990756]
0.21856702619414484
0.004482189026141918
r:[-2.276271186440681, 2.744915254237288]
[0.09090909090908939, 0.6363636363636365]
0.4159323228762778
3.101851013516526e-31
r:[1.7763568394002505e-15, 8.881784197001252e-16]
2-element Vector{Float64}:
 0.09090909090908939
 0.6363636363636365
{%endhighlight%}
Plot a quadratic curve and visualise CG steps.
{%highlight julia%}
function plot_quadf(A, b, c)
    x = -10:1:10
    y = -10:1:10
    t3 = []
    for i in x
        for j in y
            z = [i; j]
            f = 0.5*z'*A*z.-b'*z.+c
            push!(t3, f)
        end
    end
    contour(y, x, t3)
    plot!([-8, 0.961248073959938, 0.09090909090908283], [-8, -0.5687211093990756, 0.636], 2)
end

plot_quadf([4 1;1 3.], [1,2.], 0)
{%endhighlight%}

![Image 4]({{site.baseurl}}/assets/images/conjugate-gradients/cg1.svg)


## References
1. [[She94] J.R. Shewchuk. "An introduction to the conjugate gradient
method without the agonizing pain", (1994)](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

