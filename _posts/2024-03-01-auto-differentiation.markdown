---
layout: sidebar
title:  "Automatic differentiation"
date:   2024-03-01 21:16:00 +0100
categories: ML 
---

* toc
{:toc}


Automatic differentiation (AD), also known as "autodiff", is a set of methods to compute the numerical values for partial derivatives. The 2 modes of AD are 
* Forward mode auto differentiation
* Reverse mode auto differentiation 

## Composite functions
Recall the definition of the composite function (function of a function) in mathematics:
<div class="card border-primary mb-3">
    <div class="card-header">Composite function</div>   
    <div class="card-body">
Given 2 functions $f(x)$ and $g(x)$, we can compose a new function $h(x)$ when we apply function $g(x)$ over $f(x)$.
$$\begin{equation}h(x) =  g(f(x))\end{equation}$$
    </div>
</div>
If $f(x,y)$ is a defined to be an addition operation $x+y$, and $g(x,y)$ is a multiplication operation $x * y$, the process of applying $g|_{x=f}$ over $f$ describes the sequence of operation in obtaining $h(x,y) = (x + y) * y$.

Similarly, $g|_{y=f}$ describes the sequence of operation in obtaining $h(x,y) = x * (x + y)$. 
## Forward mode 
The forward mode auto differentiation (AD) evaluates the gradients in the same forward pass as the full expression evaluation, hence named "forward". The gradient of each independent variable is propagated forward with the gradient of other variables set to 0. For instance, given a set of $n$ independent variables $x_1,x_2,...x_n$, the first forward pass is initialized with $\frac{\partial{x_1}}{\partial{x_1}}=1,\frac{\partial{x_1}}{\partial{x_2}},...\frac{\partial{x_1}}{\partial{x_n}}=0$. The second forward pass is initialized with $\frac{\partial{x_2}}{\partial{x_2}}=1,\frac{\partial{x_2}}{\partial{x_1}},...\frac{\partial{x_2}}{\partial{x_n}}=0$. So with $n$ independent variables, there are $n$ forward passes.
This process is a bit like signal decomposition in signals and systems whereby a signal can be reconstructed by a summing up sines and cosines. The analogy is that if we vary a sine component while keeping all other sines/cosines constant, the resultant waveform varies in response to that sine component. Similarly, in our case when we are computing the change with respect to a particular variable ($\frac{\partial}{\partial{variable}}$), we treat other variables as constants.

<svg width="500" height="150">

<polyline points="100,100 120,80 140,100 120,80 140,60 180,100 140,60 140,30" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,60 200,60" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<polyline points="20,80 200,80" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<text x="20" y="100" font-size='10'>n=1</text>
<text x="20" y="75" font-size='10'>n=2</text>
<text x="20" y="50" font-size='10'>n=3</text>
<circle cx="100" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="180" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="30" r="5" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<polyline points="220,120 220,20 210,40 220,20 230,40" style="fill:none;stroke:black;stroke-width:2"/>
<text x="230" y="70" font-size='15'>Evaluation direction</text>
</svg>
We can think of a sequence of operations in terms of a tree structure. We start at the 
<span style="color:blue;">leaves (input)</span>, and with each operation step, we are stepping up a tree layer, until we reach the <span style="color:red;">root (output)</span>. Let $n$ be the depth (number of steps) of the tree structure. So at $n=1$ we have the input (leaf nodes) and at $n=n$ we have the output (root). 

The gradient evaluation sequence using the chain rule is kind of like the below:
$$\begin{aligned}
\frac{\partial{z}}{\partial{x}} = (\frac{\partial{z}}{\partial{h_{n-1}}}(\frac{\partial{h_{n-1}}}{\partial{h_{n-2}}}\cdots(\frac{\partial{h_2}}{\partial{h_1}}\frac{\partial{x}}{\partial{x}}))\cdots)
\end{aligned}$$ where $h_n$ are intermediate expressions and $z$ is the output expression.
Essentially, we are drilling down expressions to the most elementary component - variable, sum of 2 variables, product of 2 variables. After evaluating the resulting values, they are then used to evaluate more complicated compound expressions all the way to the final expression. This is why forward mode is also called "bottom-up" approach.  


### Example

$z = x^2 + yxy$

With symbolic differentiation,$\frac{\partial{z}}{\partial{x}} = 2x + y^2$, $\frac{\partial{z}}{\partial{y}} = 2yx$.  

With auto differentiation, the derivatives of the most elementary operations such as adds, multiplication etc are stored. The derivatives values of composite functions are then reconstructed using these elementary "base case" results and also by applying the chain rule.

For this example, define the 3 elementary derivative results for sum and multiplication.

<div class="card border-primary mb-3">
    <div class="card-header">1. Variable</div>  
    <div class="card-body">
        $$\begin{aligned}
        &f = x\\
        &\frac{\partial{f}}{\partial{x}} = 1 
        \end{aligned}$$
    </div>
</div>
<div class="card border-primary mb-3">
    <div class="card-header">2. Addition</div>  
    <div class="card-body">
        $$\begin{aligned}
        &f = a + b\\
        &\frac{\partial{f}}{\partial{x}} = \frac{\partial{a}}{\partial{x}} + \frac{\partial{b}}{\partial{x}} \\
        \\
        &f = a + b + c\\
        &\frac{\partial{f}}{\partial{x}} = \frac{\partial{a}}{\partial{x}} + \frac{\partial{b}}{\partial{x}} + \frac{\partial{c}}{\partial{x}}  
        \end{aligned}$$
    </div>
</div>
<div class="card border-primary mb-3">
    <div class="card-header">3. Multiplication</div>  
    <div class="card-body">
        $$\begin{aligned}
        &f = ab\\
        &\frac{\partial{f}}{\partial{x}} = \frac{\partial{a}}{\partial{x}}b + a\frac{\partial{b}}{\partial{x}}\\ 
        \\
        &f = abc\\
        &\frac{\partial{f}}{\partial{x}} = bc\frac{\partial{a}}{\partial{x}} + ac\frac{\partial{b}}{\partial{x}} + ab\frac{\partial{c}}{\partial{x}}  
        \end{aligned}$$
    </div>
</div>


Breaking down the expression to the following subexpressions,
$$\begin{aligned}
h_1 = x * x\\
h_2 = y * x\\
h_3 = h_2 * y\\
h_4 = h_1 + h_3\\
\end{aligned}$$

I will slightly abuse notation below. I define $\dot{h}$ to refer to the partial derivative of $h$ with respect to the current variable being worked on. And I remember that variables $x$, and $y$ are numerical values assigned as 6 & 7. 

<div class="row">
    <div class="col-sm-6">
        <div class="card">
            <div class="card-header"><h5>Partial $x$</h5></div>  
            <div class="card-body">
            $$\begin{aligned}
            \dot{x}=\frac{\partial{x}}{\partial{x}} &= 1,
            \dot{y}=\frac{\partial{y}}{\partial{x}} = 0\\
            x &= 6,y = 7 \\
            \dot{h_1} &= \dot{x}x +x\dot{x} = 12\\
            \dot{h_2} &= y\dot{x} = 7(1) = 7\\
            \dot{h_3} &= \dot{h_2}y = 7(7) = 49\\
            \dot{h_4} &= \dot{h_1}+\dot{h_3} = 12+49 = 61\\
            \end{aligned}$$
<svg width="600" height="250">

<circle cx="20" cy="30" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="20" cy="140" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<text x="15" y="35" font-size='20'>x</text>
<text x="15" y="145" font-size='20'>y</text>

<circle cx="150" cy="30" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="40" font-size='20'>*</text>

<circle cx="150" cy="140" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="150" font-size='20'>*</text>

<circle cx="225" cy="100" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="220" y="110" font-size='20'>*</text>

<circle cx="275" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="270" y="55" font-size='20'>+</text>

<polyline points="35,35 135,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,15 145,15" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="35,140 135,140" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,155 50,190 190,190 225,117" style="fill:none;stroke:black;stroke-width:2" />

<polyline points="25,45 50,90 120,90 145,125" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,130 180,90 215,90" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="240,95 260,95 270,65" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,20 260,20 270,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="290,50 370,50" style="fill:none;stroke:black;stroke-width:2" />

<circle cx="375" cy="50" r="15" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<text x="370" y="55" font-size='20'>z</text>

<text x="0" y="55" font-size='10' fill="red">(6,1)</text>
<text x="0" y="165" font-size='10' fill="red">(7,0)</text>
<text x="100" y="10" font-size='10' fill="red">(6,1)</text>
<text x="100" y="30" font-size='10' fill="red">(6,1)</text>
<text x="190" y="10" font-size='10' fill="red">(36,12)</text>
<text x="80" y="80" font-size='10' fill="red">(6,1)</text>
<text x="100" y="135" font-size='10' fill="red">(7,0)</text>
<text x="100" y="185" font-size='10' fill="red">(7,0)</text>
<text x="180" y="80" font-size='10' fill="red">(42,7)</text>
<text x="230" y="80" font-size='10' fill="red">(294,49)</text>
<text x="300" y="40" font-size='10' fill="red">(330,61)</text>
</svg>

            </div> 
        </div>
    </div>
    <div class="col-sm-6">
        <div class="card">
            <div class="card-header"><h5>Partial $y$</h5></div>  
            <div class="card-body">
            $$\begin{aligned}
            \dot{x}=\frac{\partial{x}}{\partial{y}} &= 0,
            \dot{y}=\frac{\partial{y}}{\partial{y}} = 1\\
            x &= 6,y = 7  \\
            \dot{h_1} &= 0\\
            \dot{h_2} &= \dot{y}x = 6\\
            \dot{h_3} &= \dot{h_2}y+h_2\dot{y} = 6(7)+42(1) = 84\\
            \dot{h_4} &= \dot{h_1}+\dot{h_3} = 84\\
            \end{aligned}$$

<svg width="600" height="250">

<circle cx="20" cy="30" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="20" cy="140" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<text x="15" y="35" font-size='20'>x</text>
<text x="15" y="145" font-size='20'>y</text>

<circle cx="150" cy="30" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="40" font-size='20'>*</text>

<circle cx="150" cy="140" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="150" font-size='20'>*</text>

<circle cx="225" cy="100" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="220" y="110" font-size='20'>*</text>

<circle cx="275" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="270" y="55" font-size='20'>+</text>

<polyline points="35,35 135,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,15 145,15" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="35,140 135,140" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,155 50,190 190,190 225,117" style="fill:none;stroke:black;stroke-width:2" />

<polyline points="25,45 50,90 120,90 145,125" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,130 180,90 215,90" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="240,95 260,95 270,65" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,20 260,20 270,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="290,50 370,50" style="fill:none;stroke:black;stroke-width:2" />

<circle cx="375" cy="50" r="15" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<text x="370" y="55" font-size='20'>z</text>

<text x="0" y="55" font-size='10' fill="red">(6,0)</text>
<text x="0" y="165" font-size='10' fill="red">(7,1)</text>
<text x="100" y="10" font-size='10' fill="red">(6,0)</text>
<text x="100" y="30" font-size='10' fill="red">(6,0)</text>
<text x="190" y="10" font-size='10' fill="red">(36,0)</text>
<text x="80" y="80" font-size='10' fill="red">(6,0)</text>
<text x="100" y="135" font-size='10' fill="red">(7,1)</text>
<text x="100" y="185" font-size='10' fill="red">(7,1)</text>
<text x="180" y="80" font-size='10' fill="red">(42,6)</text>
<text x="230" y="80" font-size='10' fill="red">(294,84)</text>
<text x="300" y="40" font-size='10' fill="red">(330,84)</text>
</svg>
            </div>  
        </div>
    </div>
</div>
We now verify that the results obtained are the same as when we plug in the numbers with symbolic differentiation.
$\frac{\partial{z}}{\partial{x}} = 2x + y^2=2(6) + 7^2 = 61$, $\frac{\partial{z}}{\partial{y}} = 2yx = 2(7)(6) = 84$ which is what we get from the forward mode autodiff.

### Code
The Python and Julia code implementation for the above example is as below. Note that the implementation for the Python version is based on evaluating 2 arguments, whereas for Julia is for multiple arguments. Julia implementation is a bit unique as it utilises metaprogramming. For instance, $Expr(:call,:+,a,b,c)$ constructs an expression of $a+b+c$ and calling $eval$ on that expression evaluates the sum.

<nav>
  <div class="nav nav-tabs" id="nav-tab" role="tablist">
    <button class="nav-link active" id="python-tab1" data-bs-toggle="tab" data-bs-target="#python1" type="button" role="tab" >Python</button>
    <button class="nav-link" id="julia-tab2" data-bs-toggle="tab" data-bs-target="#julia1" type="button" role="tab" >Julia</button>
  </div>
</nav>
<div class="tab-content" id="nav-tabContent">
  <div class="tab-pane fade show active" id="python1" role="tabpanel" >

  {%highlight python%}
class Expression:
    """
    Expression class which calls upon "Add" and "Mult" classes
    during addition and multiplication operations.
    """
    def __add__(a, b):
        return Add(a, b)

    def __mul__(a, b):
        return Mult(a, b)

class Var(Expression):
    """
    Variables input. Assigns value to Var during initialization.
    """
    def __init__(self, value):
        self.value = value

    #Calling evalexpr with variable as argument returns tuple with:
    #1st item: Value assigned to variable 'Var'.
    #2nd item: ∂Var/∂variable. 1 if Var equals to variable to be differentiated.
    #0 otherwise.
    def evalexpr(self, variable):
        if self == variable:
            return (self.value, 1)
        else:
            return (self.value, 0)

class Add(Expression):
    """
    Addition operation.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    #Calling evalexpr with variable as argument returns tuple with:
    #1st item: Sum of a and b.
    #2nd item: ∂a/∂variable + ∂b/∂variable. Note that there is recursive call in evalexp.
    #e.g. If type(a) == Var, Var's evalexpr is evaluated. If type(b) == Mult, 
    #Mult's evalexpr is evaluated. And the resulting subexpression is not a Var, 
    #the next evalexpr call recursively calls upon the subexpression's (Add/Mult) evalexpr 
    #until Var is encountered.
    def evalexpr(self, variable):
        a_value, a_partial = self.a.evalexpr(variable)
        b_value, b_partial = self.b.evalexpr(variable)
        return (a_value + b_value, a_partial + b_partial)

class Mult(Expression):
    """
    Multiplication operation.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    #Calling evalexpr with variable as argument returns tuple with:
    #1st item: Product of a and b
    #2nd item: Product rule: b * ∂a/∂variable + a * ∂b/∂variable. 
    #Note that there is recursive call in evalexp.
    #e.g. If type(a) == Var, Var's evalexpr is evaluated. If type(b) == Mult, 
    #Mult's evalexpr is evaluated. And the resulting subexpression is not a Var,
    #the next evalexpr call recursively calls upon the subexpression's (Add/Mult)
    #evalexpr until Var is encountered.
    def evalexpr(self, variable):
        a_value,a_partial = self.a.evalexpr(variable)
        b_value,b_partial = self.b.evalexpr(variable)
        return (a_value * b_value, b_value * a_partial + a_value * b_partial)



{%endhighlight%}

{%highlight python%}
x = Var(6)
y = Var(7)
z = x * x + y * x * y

print("(z,∂z/∂x):",z.evalexpr(x))
print("(z,∂z/∂y):",z.evalexpr(y))

#Output
#(z,∂z/∂x): (330, 61)
#(z,∂z/∂y): (330, 84)
{%endhighlight%}

  </div>
  <div class="tab-pane fade" id="julia1" role="tabpanel">
{%highlight julia%}
struct Var
    x::Real
end

function derivative(ex::Union{Expr,Var},variable::Var)
    #Base case 
    if typeof(ex) == Var
        if ex == variable
            return (ex.x,1) 
        else
            return (ex.x,0)
        end
    elseif typeof(ex) == Expr
        # Add case
        if ex.args[1] == :+     # Expr(:call,:+,a,b,..)
            v,∂v = 0,0
            #= Loops through arguments and recurses down each
            argument until base case is hit.
            =#
            for i in 2:length(ex.args)
                p,∂p = derivative(ex.args[i],variable)
                v += p     # sum of arguments
                ∂v += ∂p   # sum of partials Σ∂
            end
            return (v,∂v)

        # Multiplication case
        elseif ex.args[1] == :*  # Expr(:call,:*,a,b,..)
            v = []
            ∂v = []
            #= Loops through arguments and recurses down each
            argument until base case is hit.
            =#
            for i in 2:length(ex.args)
                p,∂p = derivative(ex.args[i],variable)
                push!(v,p)
                push!(∂v,∂p) 
            end

            a,∂a = 0,0
            a = reduce(*,v)  # product of arguments
            # computes ∂(abc)/∂variable = (bc)∂a/∂variable +
            #                             (ac)∂b/∂variable +
            #                              (ab)∂c/∂variable
            for i in 1:length(∂v)  
                ∂a += a*∂v[i]/v[i]
            end
            return (a,∂a)  
        end
    else
        return "Case Not Valid!"

    end

end
{%endhighlight%}

{%highlight julia%}
x = Var(6)
y = Var(7)
z = :($x*$x+($y*$x*$y))

println("(z,∂z/∂x):",derivative(z,x))
println("(z,∂z/∂y):",derivative(z,y))

#output
#(z,∂z/∂x):(330, 61.0)
#(z,∂z/∂y):(330, 84.0)
{%endhighlight%}
  </div>
 
</div>


Recall the definition of the Jacobian
<div class="card border-primary mb-3">
    <div class="card-header"><h5>Jacobian definition</h5></div>  
    <div class="card-body">
    $$\begin{equation}
    \boldsymbol{J}=\frac{\partial{\boldsymbol{f}}}{\partial{\boldsymbol{x}}}=\begin{bmatrix}
    \frac{\partial{f_1}}{\partial{x_1}} & \cdots & \frac{\partial{f_1}}{\partial{x_n}}\\
    \vdots & \ddots & \vdots\\
    \frac{\partial{f_n}}{\partial{x_1}} & \cdots & \frac{\partial{f_n}}{\partial{x_n}}\\
    \end{bmatrix}\\
    where \ \frac{\partial{f_i}}{\partial{x_j}} \ corresponds \ to \ the \ i^{th} \ row, \ j^{th} \ column\ entry\ of\ the\ Jacobian.
    \end{equation}$$
    </div>
</div>

Given $n$ independent input variables $x_1,x_2 \cdots x_n$ and $m$ dependent output variables $z_1,z_2 \cdots z_m$, 
we can construct the Jacobian as
$$\begin{equation}
\boldsymbol{J}=\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{x}}}=\begin{bmatrix}
\frac{\partial{z_1}}{\partial{x_1}} & \cdots & \frac{\partial{z_1}}{\partial{x_n}}\\
\vdots & \ddots & \vdots\\
\frac{\partial{z_m}}{\partial{x_1}} & \cdots & \frac{\partial{z_m}}{\partial{x_n}}\\
\end{bmatrix}\\
\end{equation}$$

One forward pass for a variable computes one column of the Jacobian. Hence, the full Jacobian can be computed in $n$ forward passes.  
The forward mode (AD) is more efficient when the number of input variables $n$ is lower than the outputs $m$. $f:\mathbb{R}^n \to \mathbb{R}^m, \ n \ll m $. The extreme case would be a one-to-many mapping, $f:\mathbb{R} \to \mathbb{R}^m$, where only 1 forward pass is required to compute gradients to all outputs.
$$\begin{equation} 
x \to \fbox{Forward AD(f(x))} \to \begin{bmatrix}
\frac{\partial{z_1}}{\partial{x}} \\ \vdots \\ \frac{\partial{z_m}}{\partial{x}}\\
\end{bmatrix}\\
\end{equation}$$

When  $n \gg m $, the reverse mode AD is preferable. 

## Reverse mode
The reverse mode auto differentiation (AD) first evaluates the full expression in the forward pass, then computes gradients in the backwards pass.
<div class="row">
    <div class="col-sm-6">
<svg width="500" height="150">

<polyline points="100,100 120,80 140,100 120,80 140,60 180,100 140,60 140,30" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,60 200,60" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<polyline points="20,80 200,80" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<text x="20" y="100" font-size='10'>n=1</text>
<text x="20" y="75" font-size='10'>n=2</text>
<text x="20" y="50" font-size='10'>n=3</text>
<circle cx="100" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="180" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="30" r="5" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<polyline points="220,120 220,20 210,40 220,20 230,40" style="fill:none;stroke:black;stroke-width:2"/>
<text x="230" y="70" font-size='15'>Evaluates numerical</text>
<text x="230" y="90" font-size='15'>values of subexpressions</text>
<text x="230" y="110" font-size='15'>(Forward step)</text>
</svg>
    </div>
     <div class="col-sm-6">
<svg width="500" height="150">

<polyline points="100,100 120,80 140,100 120,80 140,60 180,100 140,60 140,30" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,60 200,60" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<polyline points="20,80 200,80" style="fill:none;stroke:black;stroke-width:1" stroke-dasharray="5,5"/>
<text x="20" y="100" font-size='10'>n=1</text>
<text x="20" y="75" font-size='10'>n=2</text>
<text x="20" y="50" font-size='10'>n=3</text>
<circle cx="100" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="180" cy="100" r="5" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="140" cy="30" r="5" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<polyline points="220,20 220,120 210,100 220,120 230,100" style="fill:none;stroke:black;stroke-width:2"/>
<text x="230" y="70" font-size='15'>Evaluates and propagates</text>
<text x="230" y="90" font-size='15'>gradients to leaves</text>
<text x="230" y="110" font-size='15'>(Backwards step)</text>
</svg>
    </div>
</div>

Thinking in terms of a tree, we start at the 
<span style="color:blue;">leaves (input)</span>, progress all the way up to the <span style="color:red;">root (output)</span> in the forward pass. Then, in the backward pass, we evaluate gradients at the <span style="color:red;">root</span>, propagate gradients backwards down towards each of the <span style="color:blue;">leaves (input)</span>. If this reminds you of backpropagation, it is because the backpropagation algorithm is a kind of reverse mode AD.

The gradient evaluation sequence using the chain rule during the backwards step is kind of like the below:
$$\begin{aligned}
\frac{\partial{z}}{\partial{x}} = (\cdots((\frac{\partial{z}}{\partial{z}}\frac{\partial{z}}{\partial{h_{n-1}}})\frac{\partial{h_{n-1}}}{\partial{h_{n-2}}})\cdots\frac{\partial{h_2}}{\partial{x}})
\end{aligned}$$ where $h_n$ are intermediate expressions and $z$ is the output expression. The gradients are computed from the output expression (root) backwards towards the variables. This is why the reverse mode is also called “top-down” approach.

### Example

$z = x(x + y) + yxy$

With symbolic differentiation,$\frac{\partial{z}}{\partial{x}} = 2x + y + y^2$, $\frac{\partial{z}}{\partial{y}} = x + 2yx$.  

For autodiff, we first break the expression down to the following subexpressions,
$$\begin{aligned}
h_1 = x + y\\
h_2 = x * h_1\\
h_3 = x * y\\
h_4 = y * h_3\\
h_5 = h_2 + h_4\\
\end{aligned}$$

Again, I will slightly abuse notation below. I define $\bar{h_i}$ to refer to the partial derivative of output $z$ with respect to the intermediate variable $h_i$, $\bar{h_i}=\frac{\partial{z}}{\partial{h_i}}$. And I remember that variables $x$, and $y$ are numerical values assigned as 6 & 7.

<div class="row">
    <div class="col-sm-6">
        <div class="card">
            <div class="card-header"><h5>Forward step</h5></div>  
            <div class="card-body">
            $$\begin{aligned}
            x &= 6,y = 7 \\
            h_1 &= 6 + 7 = 13\\
            h_2 &= 6(13) = 78\\
            h_3 &= 6(7) = 42\\
            h_4 &= 7(42) = 294\\
            h_5 &= 78 + 294 = 372\\
            \end{aligned}$$
<svg width="600" height="250">

<circle cx="20" cy="30" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="20" cy="140" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<text x="15" y="35" font-size='20'>x</text>
<text x="15" y="145" font-size='20'>y</text>

<circle cx="150" cy="30" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="40" font-size='20'>*</text>

<circle cx="80" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="73" y="55" font-size='20'>+</text>

<circle cx="150" cy="140" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="150" font-size='20'>*</text>

<circle cx="225" cy="100" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="220" y="110" font-size='20'>*</text>

<circle cx="275" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="270" y="55" font-size='20'>+</text>

<polyline points="35,35 45,45 65,45" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="25,125 45,55 65,55" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="95,50 130,50 137,40" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,15 145,15" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="35,140 135,140" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,155 50,190 190,190 225,117" style="fill:none;stroke:black;stroke-width:2" />

<polyline points="25,45 50,90 120,90 145,125" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,130 180,90 215,90" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="240,95 260,95 270,65" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,20 260,20 270,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="290,50 370,50" style="fill:none;stroke:black;stroke-width:2" />

<circle cx="375" cy="50" r="15" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<text x="370" y="55" font-size='20'>z</text>

<text x="0" y="55" font-size='10' fill="red">v:6</text>
<text x="0" y="65" font-size='10' fill="red">∂:0</text>
<text x="0" y="165" font-size='10' fill="red">v:7</text>
<text x="0" y="175" font-size='10' fill="red">∂:0</text>
<text x="100" y="10" font-size='10' fill="red">v:6</text>
<text x="100" y="45" font-size='10' fill="red">v:13</text>
<text x="190" y="10" font-size='10' fill="red">v:78</text>
<text x="80" y="80" font-size='10' fill="red">v:6</text>
<text x="100" y="135" font-size='10' fill="red">v:7</text>
<text x="100" y="185" font-size='10' fill="red">v:7</text>
<text x="180" y="80" font-size='10' fill="red">v:42</text>
<text x="230" y="80" font-size='10' fill="red">v:294</text>
<text x="300" y="40" font-size='10' fill="red">v:372</text>
<text x="370" y="75" font-size='10' fill="red">v:372</text>
</svg>

            </div> 
        </div>
    </div>

    <div class="col-sm-6">
        <div class="card">
            <div class="card-header"><h5>Backward step</h5></div>  
            <div class="card-body">
            $$\begin{aligned}
            x &= 6,y = 7 \\
            \bar{h_5} &= \frac{\partial{z}}{\partial{z}} = 1\\
            \bar{h_4} &= \bar{h_5}\frac{\partial{h_5}}{\partial{h_4}} = 1(1) = 1\\
            \bar{h_2} &= \bar{h_5}\frac{\partial{h_5}}{\partial{h_2}} = 1(1) = 1\\
            \bar{h_3} &= \bar{h_4}\frac{\partial{h_4}}{\partial{h_3}} = 1(7) = 7\\
            \frac{\partial{z}}{\partial{y}} &= \bar{h_4}\frac{\partial{h_4}}{\partial{y}} = 1(42) = 42\\
            \frac{\partial{z}}{\partial{x}} &= \bar{h_3}\frac{\partial{h_3}}{\partial{x}} = 7(7) = 49\\
            \frac{\partial{z}}{\partial{y}} &= \bar{h_3}\frac{\partial{h_3}}{\partial{y}} = 7(6) = 42\\
            \bar{h_1} &= \bar{h_2}\frac{\partial{h_2}}{\partial{h_1}} = 1(6) = 6\\
            \frac{\partial{z}}{\partial{x}} &= \bar{h_2}\frac{\partial{h_2}}{\partial{x}} = 1(13) = 13\\
            \frac{\partial{z}}{\partial{x}} &= \bar{h_1}\frac{\partial{h_1}}{\partial{x}} = 6(1) = 6\\
            \frac{\partial{z}}{\partial{y}} &= \bar{h_1}\frac{\partial{h_1}}{\partial{y}} = 6(1) = 6\\

            Total \ \frac{\partial{z}}{\partial{x}} &= 49 + 13 + 6 = \fbox{68}\\
            Total \ \frac{\partial{z}}{\partial{y}} &= 42 + 42 + 6 = \fbox{90}\\  
            \end{aligned}$$
<svg width="600" height="250">

<circle cx="20" cy="30" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="20" cy="140" r="15" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<text x="15" y="35" font-size='20'>x</text>
<text x="15" y="145" font-size='20'>y</text>

<circle cx="150" cy="30" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="40" font-size='20'>*</text>

<circle cx="80" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="73" y="55" font-size='20'>+</text>

<circle cx="150" cy="140" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="145" y="150" font-size='20'>*</text>

<circle cx="225" cy="100" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="220" y="110" font-size='20'>*</text>

<circle cx="275" cy="50" r="15" stroke="black" stroke-width="2" fill="white"></circle>
<text x="270" y="55" font-size='20'>+</text>

<polyline points="35,35 45,45 65,45" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="25,125 45,55 65,55" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="95,50 130,50 137,40" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,15 145,15" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="35,140 135,140" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="20,155 50,190 190,190 225,117" style="fill:none;stroke:black;stroke-width:2" />

<polyline points="25,45 50,90 120,90 145,125" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,130 180,90 215,90" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="240,95 260,95 270,65" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="160,20 260,20 270,35" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="290,50 370,50" style="fill:none;stroke:black;stroke-width:2" />

<circle cx="375" cy="50" r="15" stroke="rgb(255,0,0)" stroke-width="2" fill="white"></circle>
<text x="370" y="55" font-size='20'>z</text>
<text x="370" y="75" font-size='10' fill="red">v:372</text>
<text x="370" y="85" font-size='10' fill="red">∂:1</text>
<text x="0" y="55" font-size='10' fill="red">v:6</text>
<text x="0" y="65" font-size='10' fill="red">∂:68</text>
<text x="0" y="165" font-size='10' fill="red">v:7</text>
<text x="0" y="175" font-size='10' fill="red">∂:90</text>
<text x="100" y="10" font-size='10' fill="red">∂:13</text>
<text x="100" y="45" font-size='10' fill="red">∂:6</text>
<text x="190" y="10" font-size='10' fill="red">∂:1</text>
<text x="80" y="80" font-size='10' fill="red">∂:49</text>
<text x="100" y="135" font-size='10' fill="red">∂:42</text>
<text x="100" y="185" font-size='10' fill="red">∂:42</text>
<text x="180" y="80" font-size='10' fill="red">∂:7</text>
<text x="230" y="80" font-size='10' fill="red">∂:1</text>
<text x="300" y="40" font-size='10' fill="red">∂:1</text>
<text x="45" y="40" font-size='10' fill="red">∂:6</text>
<text x="45" y="70" font-size='10' fill="red">∂:6</text>
</svg>

            </div> 
        </div>
    </div>
</div>

We now verify that the results obtained are the same as when we plug in the numbers with symbolic differentiation.
$\frac{\partial{z}}{\partial{x}} = 2x + y + y^2=2(6) + 7 +7 ^2 = 68$, $\frac{\partial{z}}{\partial{y}} = x + 2yx = 6 + 2(7)(6) = 90$ which is what we get from the reverse mode autodiff.

### Code
The Python and Julia code implementation for the above example is as below.

<nav>
  <div class="nav nav-tabs" id="nav-tab" role="tablist">
    <button class="nav-link active" id="python-tab2" data-bs-toggle="tab" data-bs-target="#python2" type="button" role="tab">Python</button>
    <button class="nav-link" id="python-tab2" data-bs-toggle="tab" data-bs-target="#julia2" type="button" role="tab">Julia</button>
  </div>
</nav>
<div class="tab-content" id="nav-tabContent">
  <div class="tab-pane fade show active" id="python2" role="tabpanel">
{%highlight python%}
class Expression:
    """
    Expression class which calls upon "Add" and "Mult" classes
    during addition and multiplication operations.
    """
    def __add__(a, b):
        return Add(a, b)

    def __mul__(a, b):
        return Mult(a, b)

class Var(Expression):
    """
    Variables input. Assigns value to Var during initialization.
    """
    def __init__(self, value):
        self.value = value
        self.partial = 0      #initialize gradient ∂output/∂Var.

    #Returns Var value.
    def eval(self):
        return self.value

    #Sums gradients from all paths and updates partial for this Var.
    def evalgrad(self, gradient):
        self.partial += gradient

class Add(Expression):
    """
    Addition operation.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = 0   #initialize sum value

    #Evaluates and returns result of add expression.
    #Eval recursively calls on eval() until base case (Var) is encountered.  
    def eval(self):
        self.value = self.a.eval() + self.b.eval()    #a + b
        return self.value

    #Recursively propagates gradient onwards to subexpressions.   
    def evalgrad(self, gradient):
        self.a.evalgrad(gradient)      #∂output/∂a = gradient * 1
        self.b.evalgrad(gradient)      #∂output/∂b = gradient * 1


class Mult(Expression):
    """
    Multiplication operation.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = 0    #initialize product value

    #Evaluates and returns result of product.
    #Eval recursively calls on eval() until base case (Var) is encountered. 
    def eval(self):
        self.value = self.a.eval() * self.b.eval()   # a * b
        return self.value

    #Recursively propagates gradient onwards to subexpressions.  
    def evalgrad(self, gradient):
        self.a.evalgrad(self.b.value * gradient)    #∂output/∂a = gradient * b
        self.b.evalgrad(self.a.value * gradient)    #∂output/∂b = gradient * a

{%endhighlight%}

{%highlight python%}
x = Var(6)
y = Var(7)
z = x * (x + y) + y * x * y

z.eval()       #evaluates z (forward propagation step)
z.evalgrad(1)  #evaluates gradients for all Vars (back propagation step)
print("z:",z.value)
print("∂z/∂x:",x.partial)
print("∂z/∂y:",y.partial)

#Output
#z: 372
#∂z/∂x: 68
#∂z/∂y: 90
{%endhighlight%}
  </div>
  <div class="tab-pane fade" id="julia2" role="tabpanel">
{%highlight julia%}
mutable struct Var
    x::Real
    partial::Real 
end

Var(x::Real) = Var(x,0)     # initialize partial = 0

mutable struct Node
    ex::Union{Expr,Var}
    val::Real
    child::Vector{Node}
end

Node(ex::Union{Expr,Var}) = Node(ex,0,[]) # initialize value = 0, child nodes = []

#Returns evaluation graph consisting of Nodes
function evaluate(ex::Union{Var,Expr})
    #Base case 
    if typeof(ex) == Var
        return Node(ex,ex.x,[])
    
    elseif typeof(ex) == Expr 
        # Add case
        if ex.args[1] == :+
            cur_n = Node(ex)   #create new Node for this expression
            for i in 2:length(ex.args) 
                n = evaluate(ex.args[i])  #recurses down to base case
                push!(cur_n.child,n)   #appends subnode as this node's child
                cur_n.val += n.val    #add subnode value to current node value
            end
            return cur_n

        # Multiplication case
        elseif ex.args[1] == :*
            cur_n = Node(ex)   #create new Node for this expression
            cur_n.val = 1
            for i in 2:length(ex.args)
                n = evaluate(ex.args[i])  #recurses down to base case
                push!(cur_n.child,n)   #appends subnode as this node's child 
                cur_n.val *= n.val  #multiply subnode value to current node value
            end
            return cur_n
        end
        
    else
        return "Case Not Valid!"
    end

end

function derivative(node::Node,gradient::Real)
    #Base case
    if typeof(node.ex) == Var
        node.ex.partial += gradient   # update gradient for this Var

    elseif typeof(node.ex) == Expr
        # Add case
        if node.ex.args[1] == :+     
            for i in 1:length(node.child)
                derivative(node.child[i],gradient)  #traverses each leaf nodes (Var) and updates gradient
            end

        # Multiplication case
        elseif node.ex.args[1] == :*  
            # current node val has been previously computed as a product of children 
            # e.g. if node has 3 subnodes a,b,c, gradient passed to subnode a = (bc)*gradient,
            # gradient passed to subnode b = (ac)*gradient, gradient passed to subnode c = (ab)*gradient, 
            for i in 1:length(node.child)  
                derivative(node.child[i],gradient * node.val / node.child[i].val) 
            end
        end

    else
        return "Case Not Valid!"
    end

end
{%endhighlight%}
{%highlight julia%}
x = Var(6)
y = Var(7)
z = :($x*($x+$y)+($y*$x*$y))

t = evaluate(z)
derivative(t,1)
println("z: ",t.val)
println("∂z/∂x: ",x.partial)
println("∂z/∂y: ",y.partial)

#output
#z: 372
#∂z/∂x: 68.0
#∂z/∂y: 90.0
{%endhighlight%}
   </div>
 
</div> 

The reverse mode (AD) is more efficient when the number of input variables $n$ is greater than the outputs $m$. $f:\mathbb{R}^n \to \mathbb{R}^m, \ n \gg m $. The extreme case would be a many-to-one mapping, $f:\mathbb{R}^n \to \mathbb{R}$.
$$\begin{equation} 
x_1,x_2 \cdots x_n \to \fbox{Reverse AD(f(x))} \to \begin{bmatrix}
\frac{\partial{z}}{\partial{x_1}} \ \cdots \ \frac{\partial{z}}{\partial{x_n}}\\
\end{bmatrix}\\
\end{equation}$$
This is the case for neural networks, where it is very common to have many input variables, but only a single output is obtained e.g. loss function, classification. Hence, this is why backpropagation is the favoured method for computing gradients in deep learning world.


## References
1. [https://en.wikipedia.org/wiki/Automatic_differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
2. [Automatic differentiation in machine learning: a survey
](https://arxiv.org/abs/1502.05767)
