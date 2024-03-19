---
layout: sidebar
title:  "Backpropagation with Attention Components"
date:   2024-01-18 21:16:00 +0100
categories: ML 
---

* toc
{:toc}

## Overview
This section explores backpropagating attention components (Query, Keys, Value, Softmax) individually. 

## Neural network recap
<svg width="1000" height="250">


<line x1="0" y1="50" x2="55" y2="50" style="stroke:black;stroke-width:2" />
<line x1="0" y1="100" x2="55" y2="100" style="stroke:black;stroke-width:2" />
<line x1="0" y1="150" x2="55" y2="150" style="stroke:black;stroke-width:2" />
<line x1="80" y1="50" x2="185" y2="30" style="stroke:black;stroke-width:2" />
<line x1="80" y1="50" x2="185" y2="80" style="stroke:black;stroke-width:2" />
<line x1="80" y1="50" x2="185" y2="130" style="stroke:black;stroke-width:2" />
<line x1="80" y1="50" x2="185" y2="180" style="stroke:black;stroke-width:2" />
<line x1="80" y1="100" x2="185" y2="30" style="stroke:black;stroke-width:2" />
<line x1="80" y1="100" x2="185" y2="80" style="stroke:black;stroke-width:2" />
<line x1="80" y1="100" x2="185" y2="130" style="stroke:black;stroke-width:2" />
<line x1="80" y1="100" x2="185" y2="180" style="stroke:black;stroke-width:2" />
<line x1="80" y1="150" x2="185" y2="30" style="stroke:black;stroke-width:2" />
<line x1="80" y1="150" x2="185" y2="80" style="stroke:black;stroke-width:2" />
<line x1="80" y1="150" x2="185" y2="130" style="stroke:black;stroke-width:2" />
<line x1="80" y1="150" x2="185" y2="180" style="stroke:black;stroke-width:2" />
<line x1="200" y1="0" x2="200" y2="20" style="stroke:black;stroke-width:2" />
<line x1="200" y1="50" x2="200" y2="70" style="stroke:black;stroke-width:2" />
<line x1="200" y1="100" x2="200" y2="120" style="stroke:black;stroke-width:2" />
<line x1="200" y1="150" x2="200" y2="170" style="stroke:black;stroke-width:2" />

<circle cx="70" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="70" cy="100" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="70" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="30" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="130" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<line x1="215" y1="30" x2="270" y2="30" style="stroke:black;stroke-width:2" />
<line x1="215" y1="80" x2="270" y2="80" style="stroke:black;stroke-width:2" />
<line x1="215" y1="130" x2="270" y2="130" style="stroke:black;stroke-width:2" />
<line x1="215" y1="180" x2="270" y2="180" style="stroke:black;stroke-width:2" />
<text x="350" y="125" fill="black" font-size="30">Equivalent to</text>
<rect x="600" y="0" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="630" y="0" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="660" y="0" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="30" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="630" y="30" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="660" y="30" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="60" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="630" y="60" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="660" y="60" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="90" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="630" y="90" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="660" y="90" width="30" height="30" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="705" y="90" fill="black" font-size="60">*</text>
<rect x="740" y="10" width="30" height="30" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="740" y="40" width="30" height="30" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="740" y="70" width="30" height="30" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="780" y="75" fill="black" font-size="50">+</text>
<rect x="820" y="0" width="30" height="30" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="820" y="30" width="30" height="30" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="820" y="60" width="30" height="30" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="820" y="90" width="30" height="30" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="860" y="75" fill="black" font-size="50">=</text>
<rect x="900" y="0" width="30" height="30" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="900" y="30" width="30" height="30" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="900" y="60" width="30" height="30" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="900" y="90" width="30" height="30" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="620" y="200" fill="black" font-size="50">W</text>
<text x="740" y="200" fill="black" font-size="50">x</text>
<text x="820" y="200" fill="black" font-size="50">b</text>
<text x="900" y="200" fill="black" font-size="50">z</text>
</svg>


The above left diagram shows a 2 layer neural network with 3 inputs nodes and 4 output nodes. This network takes a weighted sum of the 3 inputs and outputs 4 numbers, where the connections between the 1st and last layer nodes represent the weights. This network can be represented by a matrix multiplication $\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}=\boldsymbol{z}$ (1) as shown on the right. Written in full vector form,
$$\begin{equation}
\begin{bmatrix}
w_{11} \ w_{12} \ w_{13} \\ w_{21} \ w_{22} \ w_{23} \\ w_{31} \ w_{32} \ w_{33} \\ w_{41} \ w_{42} \ w_{43} 
\end{bmatrix} \ 
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 
\end{bmatrix}+
\begin{bmatrix}
b_1 \\ b_2 \\ b_3 \\ b_4
\end{bmatrix}= 
\begin{bmatrix}
z_1 \\ z_2 \\ z_3 \\ z_4
\end{bmatrix} 
\end{equation}$$

* Forward propagate step:

$$\begin{aligned}
&\boldsymbol{z} = \boldsymbol{W} \boldsymbol{x}+\boldsymbol{b} \\
\end{aligned}$$ \\

* Backpropagate step:

Let $\boldsymbol{y}$ be the 4x1 target vector, and using the L2 loss function, we can backprogate using the following computations
$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{z} - \boldsymbol{y}||^2_2 \\
&\frac{\partial{L}}{\partial{\boldsymbol{z}}}=\boldsymbol{z}-\boldsymbol{y} \\
&\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{W}}}=\boldsymbol{x}^T,\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1} \\
&\begin{bmatrix}
\frac{\partial{z_1}}{\partial{w_{11}}} \ \frac{\partial{z_1}}{\partial{w_{12}}} \ \frac{\partial{z_1}}{\partial{w_{13}}} \\
\frac{\partial{z_2}}{\partial{w_{21}}} \ \frac{\partial{z_2}}{\partial{w_{22}}} \ \frac{\partial{z_2}}{\partial{w_{23}}} \\
\frac{\partial{z_3}}{\partial{w_{31}}} \ \frac{\partial{z_3}}{\partial{w_{32}}} \ \frac{\partial{z_2}}{\partial{w_{33}}} \\
\frac{\partial{z_4}}{\partial{w_{41}}} \ \frac{\partial{z_4}}{\partial{w_{42}}} \ \frac{\partial{z_4}}{\partial{w_{43}}} \\
\end{bmatrix} =
\begin{bmatrix}
x_1 \ x_2 \ x_3 \\
x_1 \ x_2 \ x_3 \\
x_1 \ x_2 \ x_3 \\
x_1 \ x_2 \ x_3 \\
\end{bmatrix}, \ 
\begin{bmatrix}
\frac{\partial{z_1}}{\partial{b_1}} \\ \frac{\partial{z_2}}{\partial{b_2}} \\ \frac{\partial{z_3}}{\partial{b_3}} \\  \frac{\partial{z_4}}{\partial{b_4}} \\
\end{bmatrix}=
\begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\\end{bmatrix}\\
\end{aligned}$$

* Update step:

$$\begin{aligned}
&\boldsymbol{W}^{new} = \boldsymbol{W}-\eta \frac{\partial{L}}{\partial{\boldsymbol{W}}}, \
\boldsymbol{W}^{new} = \boldsymbol{W}-\eta (\boldsymbol{z}-\boldsymbol{y}) \boldsymbol{x}^T \\
&\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \frac{\partial{L}}{\partial{\boldsymbol{b}}}, \
\boldsymbol{b}^{new} = \boldsymbol{b}-\eta (\boldsymbol{z}-\boldsymbol{y}) 
\end{aligned}$$

We can simplify the expression (1) further by absorbing the bias vector $\boldsymbol{b}$ into the weight matrix $\boldsymbol{W}$ and adding a row of 1s into $\boldsymbol{x}$.


$$\begin{aligned}
\boldsymbol{W}\boldsymbol{x}&=\boldsymbol{z}\\
\begin{bmatrix}
w_{11} \ w_{12} \ w_{13} \ b_1 \\ w_{21} \ w_{22} \ w_{23} \ b_2 \\ w_{31} \ w_{32} \ w_{33} \ b_3 \\ w_{41} \ w_{42} \ w_{43} \ b_4  
\end{bmatrix} 
\begin{bmatrix}
x_{1} \\ x_{2} \\ x_{3}  \\ 1
\end{bmatrix}&=
\begin{bmatrix}
z_{1} \\ z_{2} \\ z_{3} \\ z_{4} 
\end{bmatrix}\\
\end{aligned}$$

## Multidimensional input
<svg width="1000" height="300">

<line x1="200" y1="50" x2="255" y2="50" style="stroke:black;stroke-width:2" />
<line x1="200" y1="150" x2="255" y2="150" style="stroke:black;stroke-width:2" />
<line x1="200" y1="250" x2="255" y2="250" style="stroke:green;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="50" style="stroke:green;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="150" style="stroke:green;stroke-width:2" />

<circle cx="270" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="250" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="400" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="400" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<line x1="415" y1="50" x2="470" y2="50" style="stroke:black;stroke-width:2" />
<line x1="415" y1="150" x2="470" y2="150" style="stroke:black;stroke-width:2" />

<rect x="0" y="50" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="100" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="50" y="50" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="50" y="100" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="100" y="50" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="100" y="100" width="50" height="50" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="150" width="50" height="50" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="50" y="150" width="50" height="50" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="100" y="150" width="50" height="50" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="290" y="200" fill="green" font-size="15">b1</text>
<text x="330" y="220" fill="green" font-size="15">b2</text>
<text x="200" y="240" fill="green" font-size="15">1</text>
<text x="300" y="45" fill="blue" font-size="15">w11</text>
<text x="280" y="90" fill="blue" font-size="15">w21</text>
<text x="280" y="120" fill="blue" font-size="15">w12</text>
<text x="300" y="145" fill="blue" font-size="15">w22</text>
<text x="420" y="45" fill="red" font-size="15">z11</text>
<text x="420" y="145" fill="red" font-size="15">z21</text>

<text x="200" y="140" fill="brown" font-size="15">x21</text>
<text x="200" y="40" fill="brown" font-size="15">x11</text>
<text x="20" y="180" fill="green" font-size="25">1</text>
<text x="5" y="135" fill="brown" font-size="25">x21</text>
<text x="5" y="85" fill="brown" font-size="25">x11</text>

<text x="220" y="57" fill="brown" font-size="25">></text>
<text x="220" y="157" fill="brown" font-size="25">></text>
<text x="220" y="257" fill="green" font-size="25">></text>

<text x="325" y="57" fill="blue" font-size="25">></text>
<text x="305" y="85" fill="blue" font-size="25" rotate="45">></text>
<text x="305" y="135" fill="blue" font-size="25" rotate="-45">></text>
<text x="305" y="157" fill="blue" font-size="25">></text>
<text x="325" y="217" fill="green" font-size="25" rotate="-55">></text>
<text x="325" y="180" fill="green" font-size="25" rotate="-65">></text>

<text x="440" y="57" fill="red" font-size="25">></text>
<text x="440" y="157" fill="red" font-size="25">></text>

<rect x="500" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="500" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="505" y="135" fill="red" font-size="25">z21</text>
<text x="505" y="85" fill="red" font-size="25">z11</text>


</svg>

We began the previous section by passing a column vector input $\boldsymbol{x}$ then passed it through the network above to get column vector $\boldsymbol{z}$. Now let us  extend that concept further by assuming that we have a matrix input instead of just a column vector. The figure above shows a 3x3 matrix (3 column vectors concatenated together) input x (brown) passed though the network weights (blue) and bias (green) to produce the z output (red). 

* Forward propagate step:

$$\begin{aligned}
\boldsymbol{W}\boldsymbol{x}=& \ \boldsymbol{z}\\
\begin{bmatrix}
w_{11} \ w_{12} \ b_1 \\ w_{21} \ w_{22} \ b_2 
\end{bmatrix} 
\begin{bmatrix}
x_{11} \ x_{12} \ x_{13} \\ x_{21} \ x_{22} \ x_{23} \\ 1 \ \ \ 1 \ \ \ 1
\end{bmatrix}=&
\begin{bmatrix}
z_{11} \ z_{12} \ z_{13} \\ z_{21} \ z_{22} \ z_{23} 
\end{bmatrix}\\
\begin{bmatrix}
w_{11}x_{11}+w_{12}x_{21}+b_1 \ \ \ w_{11}x_{12}+w_{12}x_{22}+b_1 \ \ \ w_{11}x_{13}+w_{12}x_{23}+b_1 \\
w_{21}x_{11}+w_{22}x_{21}+b_2 \ \ \ w_{21}x_{12}+w_{22}x_{22}+b_2 \ \ \ w_{21}x_{13}+w_{22}x_{23}+b_2 
\end{bmatrix}=&
\begin{bmatrix} 
z_{11} \ z_{12} \ z_{13} \\ z_{21} \ z_{22} \ z_{23} 
\end{bmatrix} 
\end{aligned}$$\\
The matrix multiplication operation can be thought of as passing 3 column vector inputs separately to the network and then concatenating the individual outputs next to each other. 

* Backpropagate step:

Let $\boldsymbol{y}$ be the 2x3 target vector, and using the L2 loss function, we calculate the error of a 2x3  $\boldsymbol{z}$ . We know how to do backpropagation with a column vector (previous section) so let us split the problem down to 3 steps - computing backpropagation with $\boldsymbol{z_1}$, then $\boldsymbol{z_2}$ then $\boldsymbol{z_3}$.

$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{z} - \boldsymbol{y}||^2_2 \\
&\frac{\partial{L}}{\partial{\boldsymbol{z}}}=\boldsymbol{z}-\boldsymbol{y} \\
\end{aligned}$$

<b>Step 1</b>
<svg width="1000" height="300">

<line x1="200" y1="50" x2="255" y2="50" style="stroke:black;stroke-width:2" />
<line x1="200" y1="150" x2="255" y2="150" style="stroke:black;stroke-width:2" />
<line x1="200" y1="250" x2="255" y2="250" style="stroke:green;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="50" style="stroke:green;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="150" style="stroke:green;stroke-width:2" />

<circle cx="270" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="250" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="400" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="400" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<line x1="415" y1="50" x2="470" y2="50" style="stroke:black;stroke-width:2" />
<line x1="415" y1="150" x2="470" y2="150" style="stroke:black;stroke-width:2" />


<text x="270" y="200" fill="green" font-size="15">∂z11</text>
<text x="330" y="220" fill="green" font-size="15">∂z21</text>

<text x="300" y="45" fill="blue" font-size="15">x11∂z11</text>
<text x="250" y="90" fill="blue" font-size="15">x11∂z21</text>
<text x="250" y="120" fill="blue" font-size="15">x21∂z11</text>
<text x="300" y="145" fill="blue" font-size="15">x21∂z21</text>
<text x="420" y="45" fill="red" font-size="15">∂z11</text>
<text x="420" y="145" fill="red" font-size="15">∂z21</text>



<text x="325" y="57" fill="blue" font-size="25"><</text>
<text x="305" y="85" fill="blue" font-size="25" rotate="45"><</text>
<text x="305" y="135" fill="blue" font-size="25" rotate="-45"><</text>
<text x="305" y="157" fill="blue" font-size="25"><</text>
<text x="325" y="217" fill="green" font-size="25" rotate="-45"><</text>
<text x="325" y="180" fill="green" font-size="25" rotate="-65"><</text>

<text x="440" y="57" fill="red" font-size="25"><</text>
<text x="440" y="157" fill="red" font-size="25"><</text>

<rect x="500" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="500" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="505" y="135" fill="black" font-size="15">∂z21</text>
<text x="505" y="85" fill="black" font-size="15">∂z11</text>
<text x="555" y="135" fill="black" font-size="15">∂z22</text>
<text x="555" y="85" fill="black" font-size="15">∂z12</text>
<text x="605" y="135" fill="black" font-size="15">∂z23</text>
<text x="605" y="85" fill="black" font-size="15">∂z13</text>

</svg>

* Step 1 - Backpropagate step:

$$\begin{aligned}
&\frac{\partial{\boldsymbol{z_1}}}{\partial{\boldsymbol{W}}}=\boldsymbol{x_1}^T,\frac{\partial{\boldsymbol{z_1}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1} \\
&\begin{bmatrix}
\frac{\partial{z_{11}}}{\partial{w_{11}}} \ \frac{\partial{z_{11}}}{\partial{w_{12}}}  \\
\frac{\partial{z_{21}}}{\partial{w_{21}}} \ \frac{\partial{z_{21}}}{\partial{w_{22}}}  \\
\end{bmatrix} =
\begin{bmatrix}
x_{11} \ x_{21} \\
x_{11} \ x_{21} \\
\end{bmatrix}, \ 
\begin{bmatrix}
\frac{\partial{z_{11}}}{\partial{b_1}} \\ \frac{\partial{z_{21}}}{\partial{b_2}} \\
\end{bmatrix}=
\begin{bmatrix} 1 \\ 1 \\\end{bmatrix}\\
\end{aligned}$$

* Step 1 - Update step:

$$\begin{aligned}
&\boldsymbol{W}^{new} = \boldsymbol{W}-\eta \frac{\partial{L}}{\partial{\boldsymbol{W}}}, \
\boldsymbol{W}^{new} = \boldsymbol{W}-\eta (\boldsymbol{z_1}-\boldsymbol{y_1}) \boldsymbol{x_1}^T \\
&\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \frac{\partial{L}}{\partial{\boldsymbol{b}}}, \
\boldsymbol{b}^{new} = \boldsymbol{b}-\eta (\boldsymbol{z_1}-\boldsymbol{y_1}) 
\end{aligned}$$

<b>Step 2</b>
<svg width="1000" height="300">

<line x1="200" y1="50" x2="255" y2="50" style="stroke:black;stroke-width:2" />
<line x1="200" y1="150" x2="255" y2="150" style="stroke:black;stroke-width:2" />
<line x1="200" y1="250" x2="255" y2="250" style="stroke:green;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="50" style="stroke:green;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="150" style="stroke:green;stroke-width:2" />

<circle cx="270" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="250" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="400" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="400" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<line x1="415" y1="50" x2="470" y2="50" style="stroke:black;stroke-width:2" />
<line x1="415" y1="150" x2="470" y2="150" style="stroke:black;stroke-width:2" />


<text x="270" y="200" fill="green" font-size="15">∂z12</text>
<text x="330" y="220" fill="green" font-size="15">∂z22</text>

<text x="300" y="45" fill="blue" font-size="15">x12∂z12</text>
<text x="250" y="90" fill="blue" font-size="15">x12∂z22</text>
<text x="250" y="120" fill="blue" font-size="15">x22∂z12</text>
<text x="300" y="145" fill="blue" font-size="15">x22∂z22</text>
<text x="420" y="45" fill="red" font-size="15">∂z12</text>
<text x="420" y="145" fill="red" font-size="15">∂z22</text>



<text x="325" y="57" fill="blue" font-size="25"><</text>
<text x="305" y="85" fill="blue" font-size="25" rotate="45"><</text>
<text x="305" y="135" fill="blue" font-size="25" rotate="-45"><</text>
<text x="305" y="157" fill="blue" font-size="25"><</text>
<text x="325" y="217" fill="green" font-size="25" rotate="-45"><</text>
<text x="325" y="180" fill="green" font-size="25" rotate="-65"><</text>

<text x="440" y="57" fill="red" font-size="25"><</text>
<text x="440" y="157" fill="red" font-size="25"><</text>

<rect x="500" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="500" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="505" y="135" fill="black" font-size="15">∂z21</text>
<text x="505" y="85" fill="black" font-size="15">∂z11</text>
<text x="555" y="135" fill="black" font-size="15">∂z22</text>
<text x="555" y="85" fill="black" font-size="15">∂z12</text>
<text x="605" y="135" fill="black" font-size="15">∂z23</text>
<text x="605" y="85" fill="black" font-size="15">∂z13</text>

</svg>

* Step 2 - Backpropagate step:

$$\begin{aligned}
&\frac{\partial{\boldsymbol{z_2}}}{\partial{\boldsymbol{W}}}=\boldsymbol{x_2}^T,\frac{\partial{\boldsymbol{z_2}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1} \\
&\begin{bmatrix}
\frac{\partial{z_{12}}}{\partial{w_{11}}} \ \frac{\partial{z_{12}}}{\partial{w_{12}}}  \\
\frac{\partial{z_{22}}}{\partial{w_{21}}} \ \frac{\partial{z_{21}}}{\partial{w_{22}}}  \\
\end{bmatrix} =
\begin{bmatrix}
x_{12} \ x_{22} \\
x_{12} \ x_{22} \\
\end{bmatrix}, \ 
\begin{bmatrix}
\frac{\partial{z_{12}}}{\partial{b_1}} \\ \frac{\partial{z_{22}}}{\partial{b_2}} \\
\end{bmatrix}=
\begin{bmatrix} 1 \\ 1 \\\end{bmatrix}\\
\end{aligned}$$

* Step 2 - Update step:

$$\begin{aligned}
&\boldsymbol{W}^{new} = \boldsymbol{W}-\eta \frac{\partial{L}}{\partial{\boldsymbol{W}}}, \
\boldsymbol{W}^{new} = \boldsymbol{W}-\eta (\boldsymbol{z_2}-\boldsymbol{y_2}) \boldsymbol{x_2}^T \\
&\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \frac{\partial{L}}{\partial{\boldsymbol{b}}}, \
\boldsymbol{b}^{new} = \boldsymbol{b}-\eta (\boldsymbol{z_2}-\boldsymbol{y_2}) 
\end{aligned}$$

<b>Step 3</b>
<svg width="1000" height="300">

<line x1="200" y1="50" x2="255" y2="50" style="stroke:black;stroke-width:2" />
<line x1="200" y1="150" x2="255" y2="150" style="stroke:black;stroke-width:2" />
<line x1="200" y1="250" x2="255" y2="250" style="stroke:green;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="50" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="50" style="stroke:black;stroke-width:2" />
<line x1="280" y1="150" x2="385" y2="150" style="stroke:black;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="50" style="stroke:green;stroke-width:2" />
<line x1="280" y1="250" x2="385" y2="150" style="stroke:green;stroke-width:2" />

<circle cx="270" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="270" cy="250" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="400" cy="50" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="400" cy="150" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<line x1="415" y1="50" x2="470" y2="50" style="stroke:black;stroke-width:2" />
<line x1="415" y1="150" x2="470" y2="150" style="stroke:black;stroke-width:2" />


<text x="270" y="200" fill="green" font-size="15">∂z13</text>
<text x="330" y="220" fill="green" font-size="15">∂z23</text>

<text x="300" y="45" fill="blue" font-size="15">x13∂z13</text>
<text x="250" y="90" fill="blue" font-size="15">x13∂z23</text>
<text x="250" y="120" fill="blue" font-size="15">x23∂z13</text>
<text x="300" y="145" fill="blue" font-size="15">x23∂z23</text>
<text x="420" y="45" fill="red" font-size="15">∂z13</text>
<text x="420" y="145" fill="red" font-size="15">∂z23</text>



<text x="325" y="57" fill="blue" font-size="25"><</text>
<text x="305" y="85" fill="blue" font-size="25" rotate="45"><</text>
<text x="305" y="135" fill="blue" font-size="25" rotate="-45"><</text>
<text x="305" y="157" fill="blue" font-size="25"><</text>
<text x="325" y="217" fill="green" font-size="25" rotate="-45"><</text>
<text x="325" y="180" fill="green" font-size="25" rotate="-65"><</text>

<text x="440" y="57" fill="red" font-size="25"><</text>
<text x="440" y="157" fill="red" font-size="25"><</text>

<rect x="500" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="500" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="50" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="100" width="50" height="50" style="fill:rgb(100,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="50" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="600" y="100" width="50" height="50" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="505" y="135" fill="black" font-size="15">∂z21</text>
<text x="505" y="85" fill="black" font-size="15">∂z11</text>
<text x="555" y="135" fill="black" font-size="15">∂z22</text>
<text x="555" y="85" fill="black" font-size="15">∂z12</text>
<text x="605" y="135" fill="black" font-size="15">∂z23</text>
<text x="605" y="85" fill="black" font-size="15">∂z13</text>

</svg>

* Step 3 - Backpropagate step:

$$\begin{aligned}
&\frac{\partial{\boldsymbol{z_3}}}{\partial{\boldsymbol{W}}}=\boldsymbol{x_3}^T,\frac{\partial{\boldsymbol{z_3}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1} \\
&\begin{bmatrix}
\frac{\partial{z_{13}}}{\partial{w_{11}}} \ \frac{\partial{z_{13}}}{\partial{w_{12}}}  \\
\frac{\partial{z_{23}}}{\partial{w_{21}}} \ \frac{\partial{z_{23}}}{\partial{w_{22}}}  \\
\end{bmatrix} =
\begin{bmatrix}
x_{13} \ x_{23} \\
x_{13} \ x_{23} \\
\end{bmatrix}, \ 
\begin{bmatrix}
\frac{\partial{z_{13}}}{\partial{b_1}} \\ \frac{\partial{z_{23}}}{\partial{b_2}} \\
\end{bmatrix}=
\begin{bmatrix} 1 \\ 1 \\\end{bmatrix}\\
\end{aligned}$$

* Step 3 - Update step:

$$\begin{aligned}
&\boldsymbol{W}^{new} = \boldsymbol{W}-\eta \frac{\partial{L}}{\partial{\boldsymbol{W}}}, \
\boldsymbol{W}^{new} = \boldsymbol{W}-\eta (\boldsymbol{z_3}-\boldsymbol{y_3}) \boldsymbol{x_3}^T \\
&\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \frac{\partial{L}}{\partial{\boldsymbol{b}}}, \
\boldsymbol{b}^{new} = \boldsymbol{b}-\eta (\boldsymbol{z_3}-\boldsymbol{y_3}) 
\end{aligned}$$

If we stare at it closely, the 3 steps are like updating the $\boldsymbol{W}$ and the bias parameters 3 times. So we can actually combine all 3 steps into a matrix multiplication step.
$$\begin{aligned}
\begin{bmatrix}
\frac{\partial{L}}{\partial{z_{11}}} \ \frac{\partial{L}}{\partial{z_{12}}} \ \frac{\partial{L}}{\partial{z_{13}}}\\
\frac{\partial{L}}{\partial{z_{21}}} \ \frac{\partial{L}}{\partial{z_{22}}} \ \frac{\partial{L}}{\partial{z_{23}}}
\end{bmatrix}
\begin{bmatrix}
x_{11} \ x_{21} \\
x_{12} \ x_{22} \\
x_{13} \ x_{23} \\
\end{bmatrix}=
\begin{bmatrix}
\frac{\partial{L}}{\partial{z_{11}}}x_{11}+\frac{\partial{L}}{\partial{z_{12}}}x_{12}+\frac{\partial{L}}{\partial{z_{13}}}x_{13} \ \ \
\frac{\partial{L}}{\partial{z_{11}}}x_{21}+\frac{\partial{L}}{\partial{z_{12}}}x_{22}+\frac{\partial{L}}{\partial{z_{13}}}x_{23} \\
\frac{\partial{L}}{\partial{z_{21}}}x_{11}+\frac{\partial{L}}{\partial{z_{22}}}x_{12}+\frac{\partial{L}}{\partial{z_{23}}}x_{13} \ \ \
\frac{\partial{L}}{\partial{z_{21}}}x_{21}+\frac{\partial{L}}{\partial{z_{22}}}x_{22}+\frac{\partial{L}}{\partial{z_{23}}}x_{23} \\  
\end{bmatrix}
\end{aligned}$$

* Update step:

$$\begin{aligned}
&\boldsymbol{W}^{new} = \boldsymbol{W}-\eta \frac{\partial{L}}{\partial{\boldsymbol{W}}}, \
\boldsymbol{W}^{new} = \boldsymbol{W}-\eta (\boldsymbol{z}-\boldsymbol{y}) \boldsymbol{x}^T \\
&\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \frac{\partial{L}}{\partial{\boldsymbol{b}}}, \
\boldsymbol{b}^{new} = \boldsymbol{b}-\eta \ sumcols(\boldsymbol{z}-\boldsymbol{y}) 
\end{aligned}$$
where sumcols means summing the columns.

## Values, V
<svg width="800" height="300">

<text x="0" y="57" fill="black" font-size="15">α1</text>
<polyline points="0,60 55,60 " style="fill:none;stroke:black;stroke-width:1" />
<text x="0" y="97" fill="black" font-size="15">v1</text>
<polyline points="0,100 40,100 60 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="70" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="65" y="72" fill="black" font-size="25">*</text>
<text x="100" y="57" fill="black" font-size="15">α1.v1</text>
<polyline points="85,60 150,60 200,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="215" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="207" y="125" fill="black" font-size="25">+</text>
<polyline points="230,117 340,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="240" y="110" fill="black" font-size="15">z=α1.v1+α2.v2</text>
<circle cx="70" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="100" y="177" fill="black" font-size="15">α2.v2</text>
<text x="65" y="192" fill="black" font-size="25">*</text>
<text x="0" y="177" fill="black" font-size="15">α2</text>
<polyline points="0,180 55,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="0" y="217" fill="black" font-size="15">v2</text>
<polyline points="85,180 150,180 200,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="0,220 40,220 60 190" style="fill:none;stroke:black;stroke-width:1" />
</svg>

* Forward propagate step:

$$
\boldsymbol{v_1} =\begin{bmatrix}
 v_{11} \\ v_{21} \\ v_{31} 
\end{bmatrix}, \
\boldsymbol{v_2} = \begin{bmatrix}
v_{12} \\ v_{22} \\ v_{32} 
\end{bmatrix}, \
\boldsymbol{\alpha} = \begin{bmatrix}
\alpha_{1} \ \alpha_{2} \ 
\end{bmatrix}\\
$$

Concatenating $\boldsymbol{v_1}$ and $\boldsymbol{v_2}$ to $\boldsymbol{v}$ 
$$\begin{aligned}
\boldsymbol{v} = \begin{bmatrix}
v_{11} \  v_{12} \\ v_{21} \ v_{22} \\ v_{31} \ v_{32}  
\end{bmatrix} 
\end{aligned}$$

$\boldsymbol{z}$ can be computed vectorially as
$$\begin{aligned}
&\boldsymbol{z} = \begin{bmatrix}
v_{11} \  v_{12} \\ v_{21} \ v_{22} \\ v_{31} \ v_{32} 
\end{bmatrix}
\begin{bmatrix}
\alpha_{1} \\ \alpha_{2}
\end{bmatrix}\\
&\boldsymbol{z} =  \boldsymbol{v} \boldsymbol{\alpha}^T 
\end{aligned}$$

* Backprogate step:

Let $\boldsymbol{y}$ be the 3x1 target vector.

$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{z} - \boldsymbol{y}||^2_2 \\
&\frac{\partial{L}}{\partial{\boldsymbol{z}}}=\boldsymbol{z}-\boldsymbol{y} \\
&\frac{\partial{L}}{\partial{\boldsymbol{z}}}=\begin{bmatrix}
z_1 - y_1 \\
z_2 - y_2 \\
z_3 - y_3 \\
\end{bmatrix}\\
&\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{v}}}=\boldsymbol{\alpha}\\
&\begin{bmatrix}
\frac{\partial{z_1}}{\partial{v_{11}}} \ \frac{\partial{z_1}}{\partial{v_{12}}}  \\
\frac{\partial{z_2}}{\partial{v_{21}}} \ \frac{\partial{z_2}}{\partial{v_{22}}}  \\
\frac{\partial{z_3}}{\partial{v_{31}}} \ \frac{\partial{z_3}}{\partial{v_{32}}}  \\
\end{bmatrix} =
\begin{bmatrix}
\alpha_{1} \ \alpha_{2} \\
\alpha_{1} \ \alpha_{2} \\
\alpha_{1} \ \alpha_{2} \\
\end{bmatrix}, \\ 

&\frac{\partial{L}}{\partial{\boldsymbol{v}}}=\frac{\partial{L}}{\partial{\boldsymbol{z}}}\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{v}}}\\


\end{aligned}$$

<svg width="500" height="300">

<text x="0" y="57" fill="black" font-size="15">α1</text>
<polyline points="0,60 55,60 " style="fill:none;stroke:black;stroke-width:1" />
<text x="0" y="97" fill="black" font-size="15">v1</text>
<polyline points="0,100 40,100 60 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="70" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="65" y="72" fill="black" font-size="25">*</text>
<text x="100" y="57" fill="black" font-size="15">α1.v1</text>
<polyline points="85,60 150,60 200,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="215" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="207" y="125" fill="black" font-size="25">+</text>
<polyline points="230,117 340,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="240" y="110" fill="black" font-size="15">z=α1.v1+α2.v2</text>
<circle cx="70" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="100" y="177" fill="black" font-size="15">α2.v2</text>
<text x="65" y="192" fill="black" font-size="25">*</text>
<text x="0" y="177" fill="black" font-size="15">α2</text>
<polyline points="0,180 55,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="0" y="217" fill="black" font-size="15">v2</text>
<polyline points="85,180 150,180 200,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="0,220 40,220 60 190" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="270,120 260,115 270 110" style="fill:none;stroke:red;stroke-width:3" />
<text x="240" y="140" fill="red" font-size="15">∂L/∂z</text>
<polyline points="140,65 130,60 140 55" style="fill:none;stroke:red;stroke-width:3" />
<text x="100" y="80" fill="red" font-size="15">∂L/∂z</text>
<polyline points="140,185 130,180 140 175" style="fill:none;stroke:red;stroke-width:3" />
<text x="100" y="200" fill="red" font-size="15">∂L/∂z</text>
<polyline points="20,105 10,100 20 95" style="fill:none;stroke:red;stroke-width:3" />
<text x="10" y="120" fill="red" font-size="15">α1.∂L/∂z</text>
<polyline points="20,225 10,220 20 215" style="fill:none;stroke:red;stroke-width:3" />
<text x="10" y="240" fill="red" font-size="15">α2.∂L/∂z</text>
</svg>


<svg width="300" height="300">

<rect x="0" y="0" width="30" height="30" style="fill:rgb(0,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="30" width="30" height="30" style="fill:rgb(0,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="60" width="30" height="30" style="fill:rgb(0,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="40" y="50" fill="black" font-size="25">*</text>
<rect x="60" y="0" width="30" height="30" style="fill:rgb(100,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="90" y="0" width="30" height="30" style="fill:rgb(100,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="130" y="50" fill="black" font-size="25">=</text>
<rect x="150" y="0" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="180" y="0" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="150" y="0" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="180" y="0" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="150" y="30" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="180" y="30" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="150" y="60" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="180" y="60" width="30" height="30" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="0" y="140" fill="black" font-size="25" font-weight="bold">z-y</text>
<text x="80" y="140" fill="black" font-size="25" font-weight="bold">α</text>
<text x="150" y="140" fill="black" font-size="25" font-weight="bold">(z-y)α</text>
</svg>

Intuitively from the left diagram, we can think of the errors being backpropagated to $v_1$ and $v_2$ by the scaling factors $\alpha_1$, $\alpha_2$ respectively.

* Update step:
$$\begin{aligned}
&\boldsymbol{v}^{new}=\boldsymbol{v}-\eta\frac{\partial{L}}{\partial{\boldsymbol{v}}}\\
&\boldsymbol{v}^{new}=\boldsymbol{v}-\eta(\boldsymbol{z}-\boldsymbol{y})\boldsymbol{\alpha}\\
\end{aligned}$$

Next let us try to start stacking layers. For example, instead of $v_1$ & $v_2$ being the trainable parameters, they could be the output of a 2 layer feedforward network as shown below.  

<svg width="1000" height="250">


<line x1="50" y1="50" x2="100" y2="50" style="stroke:black;stroke-width:2" />
<line x1="50" y1="80" x2="100" y2="80" style="stroke:black;stroke-width:2" />
<line x1="50" y1="110" x2="100" y2="110" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="80" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="110" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="80" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="110" style="stroke:black;stroke-width:2" />
<line x1="110" y1="110" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="110" x2="190" y2="80" style="stroke:black;stroke-width:2" />
<line x1="110" y1="110" x2="190" y2="110" style="stroke:black;stroke-width:2" />


<circle cx="100" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="110" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="110" r="10" stroke="black" stroke-width="1" fill="white"></circle>


<line x1="210" y1="50" x2="270" y2="50" style="stroke:black;stroke-width:2" />
<line x1="210" y1="80" x2="270" y2="80" style="stroke:black;stroke-width:2" />
<line x1="210" y1="110" x2="270" y2="110" style="stroke:black;stroke-width:2" />

<rect x="0" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="0" y="90" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="25" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="25" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="25" y="90" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="90" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="265" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="265" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="265" y="90" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="5" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="30" y="30" fill="black" font-size="15" font-weight="bold">x2</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">v1</text>
<text x="270" y="30" fill="black" font-size="15" font-weight="bold">v2</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">V weights</text>

<text x="400" y="57" fill="black" font-size="15">α1</text>
<polyline points="400,60 455,60 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">v1</text>
<polyline points="250,115 250,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">α1.v1</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">z=α1.v1+α2.v2</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">α2.v2</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">α2</text>
<polyline points="400,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">v2</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="280,115 280,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />


</svg>

<svg width="1000" height="250">

<text x="0" y="95" fill="black" font-size="50">(</text>
<rect x="20" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="20" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="20" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="45" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="45" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="45" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="70" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="70" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="70" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="110" y="105" fill="black" font-size="50">*</text>
<text x="30" y="170" fill="black" font-size="50" font-weight="bold">V</text>

<rect x="140" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="140" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="140" y="90" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="165" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="165" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="165" y="90" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="205" y="95" fill="black" font-size="40">+</text>
<text x="160" y="170" fill="black" font-size="50" font-weight="bold">x</text>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="285" y="95" fill="black" font-size="50">)</text>
<text x="310" y="105" fill="black" font-size="50">*</text>
<text x="240" y="170" fill="black" font-size="50" font-weight="bold">b</text>

<rect x="350" y="40" width="25" height="25" style="fill:rgb(100,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="350" y="65" width="25" height="25" style="fill:rgb(100,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="400" y="105" fill="black" font-size="50">=</text>
<text x="350" y="170" fill="black" font-size="50" 
font-weight="bold">α'</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(200,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="65" width="25" height="25" style="fill:rgb(200,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="90" width="25" height="25" style="fill:rgb(200,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="450" y="170" fill="black" font-size="50" font-weight="bold">z</text>
</svg>

In this case we have the matrix $\boldsymbol{V}$ as the trainable parameters and
$\boldsymbol{v}=\boldsymbol{V}\boldsymbol{x}+\boldsymbol{b}$
* Backprogate step:

$$\begin{aligned}

&\frac{\partial{\boldsymbol{v}}}{\partial{\boldsymbol{V}}}=\boldsymbol{x}^T,
&\frac{\partial{\boldsymbol{v}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1}\\

&\frac{\partial{L}}{\partial{\boldsymbol{V}}}=\frac{\partial{L}}{\partial{\boldsymbol{z}}}\frac{\partial{\boldsymbol{z}}}{\partial{\boldsymbol{v}}}\frac{\partial{\boldsymbol{v}}}{\partial{\boldsymbol{V}}}\\


&\frac{\partial{L}}{\partial{\boldsymbol{V}}}=
\begin{bmatrix}
\frac{\partial{L}}{\partial{v_{11}}} \ \frac{\partial{L}}{\partial{v_{12}}}  \\
\frac{\partial{L}}{\partial{v_{21}}} \ \frac{\partial{L}}{\partial{v_{22}}}  \\
\frac{\partial{L}}{\partial{v_{31}}} \ \frac{\partial{L}}{\partial{v_{32}}}  \\
\end{bmatrix} *
\begin{bmatrix}
x_{11} \ x_{21} \ x_{31}\\
x_{12} \ x_{22} \ x_{32}\\
\end{bmatrix} \\ 
&= 
\begin{bmatrix}
\frac{\partial{L}}{\partial{v_{11}}}x_{11}+\frac{\partial{L}}{\partial{v_{12}}}x_{12} \ \ \frac{\partial{L}}{\partial{v_{11}}}x_{21}+\frac{\partial{L}}{\partial{v_{12}}}x_{22} \ \ \frac{\partial{L}}{\partial{v_{11}}}x_{31}+\frac{\partial{L}}{\partial{v_{12}}}x_{32} \\
\frac{\partial{L}}{\partial{v_{21}}}x_{11}+\frac{\partial{L}}{\partial{v_{22}}}x_{12} \ \ \frac{\partial{L}}{\partial{v_{21}}}x_{21}+\frac{\partial{L}}{\partial{v_{22}}}x_{22} \ \ \frac{\partial{L}}{\partial{v_{21}}}x_{31}+\frac{\partial{L}}{\partial{v_{22}}}x_{32} \\
\frac{\partial{L}}{\partial{v_{31}}}x_{11}+\frac{\partial{L}}{\partial{v_{32}}}x_{12} \ \ \frac{\partial{L}}{\partial{v_{31}}}x_{21}+\frac{\partial{L}}{\partial{v_{32}}}x_{22} \ \ \frac{\partial{L}}{\partial{v_{31}}}x_{31}+\frac{\partial{L}}{\partial{v_{32}}}x_{32} \\
\end{bmatrix} 

\end{aligned}$$

* Update step:

$$\begin{aligned}
&\boldsymbol{V}^{new}=\boldsymbol{V}-\eta\frac{\partial{L}}{\partial{\boldsymbol{V}}}\\
&\boldsymbol{V}^{new}=\boldsymbol{V}-\eta(\boldsymbol{z}-\boldsymbol{y})\boldsymbol{\alpha}\boldsymbol{x}^T\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="200" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="250" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="250" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="250" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="210" y="140" fill="black" font-size="25" font-weight="bold">Vnew</text>
<text x="280" y="90" fill="black" font-size="25">=</text>
<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="350" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="350" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="350" y="90" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="335" y="140" fill="black" font-size="25" font-weight="bold">V</text>
<text x="390" y="90" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="90" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="90" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="440" y="140" fill="black" font-size="25" font-weight="bold">(z-y)α</text>

<rect x="540" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="540" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="590" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="590" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="510" y="95" fill="black" font-size="25" font-weight="bold">*</text>
<text x="560" y="140" fill="black" font-size="25" font-weight="bold">x'</text>

<text x="620" y="90" fill="black" font-size="25" font-weight="bold">=</text>
<rect x="650" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="650" y="65" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="650" y="90" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="65" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="90" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="700" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="700" y="65" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="700" y="90" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="650" y="140" fill="black" font-size="25" font-weight="bold">(z-y)αx'</text>
<text x="750" y="90" fill="black" font-size="25" font-weight="bold">)</text>
</svg>

Notice that the product of the purple and brown block has an implied sum of the change in $\boldsymbol{V}$ of both $v_1$, $v_2$. We can think of it as similar to combining 2 independent sample 'iteration' updates (pass $x_1$ in iteration 1 and then updating $\boldsymbol{V}$. Pass $x_2$ in iteration 2 and then updating $\boldsymbol{V}$) 


$$\begin{aligned}
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta\frac{\partial{L}}{\partial{\boldsymbol{b}}}\\
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta \ sumcols((\boldsymbol{z}-\boldsymbol{y})\boldsymbol{\alpha})\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="240" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="220" y="140" fill="black" font-size="25" font-weight="bold">bnew</text>
<text x="290" y="90" fill="black" font-size="25">=</text>
<rect x="330" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="330" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="330" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="335" y="140" fill="black" font-size="25" font-weight="bold">b</text>
<text x="390" y="90" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="90" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="90" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="440" y="140" fill="black" font-size="25" font-weight="bold">(z-y)α</text>
<polyline points="510,35 400,35 430,25 400,35 430,45 " style="fill:none;stroke:black;stroke-width:5" />
<text x="430" y="20" fill="black" font-size="25" font-weight="bold">sum cols</text>
<text x="520" y="90" fill="black" font-size="25">=</text>
<rect x="550" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="90" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="600" y="90" fill="black" font-size="25" font-weight="bold">)</text>
</svg>



To update the bias parameter, we will need to sum the 2 columns of $\frac{\partial{L}}{\partial{\boldsymbol{v}}}$. Reason being the bias is shared for both $x_1$, $x_2$. Think of it as combining 2 sample 'iteration' updates (pass $x_1$ in iteration 1 and then updating $\boldsymbol{V}$, $b$. Pass $x_2$ in iteration 2 and then updating $\boldsymbol{V}$, $b$).


The following julia code illustrates an example.

{% highlight julia %}

#L2 norm
function L2norm(x)
    sqrt(sum(x.^2)) 
end

#Squared Loss
function Squared_Loss(z,x)
    return 0.5*(L2norm(z-x))^2
end

#Computes output z
function FeedForward(W,b,x,y,α)
    v = W*x.+b
    z = v * α'
    return z
end

# Forward propagate
function forwardprop(W,b,x,y,α)
    z = FeedForward(W,b,x,y,α)
    return backprop(W,b,x,z,y,α)
end

# Backpropate
function backprop(W,b,x,z,y,α,η=0.002)
    println("Loss:",Squared_Loss(z,y))
    ∂L_∂z = z-y
    ∂z_∂v = α
    ∂v_∂w = x'

    ∂L_∂w = ∂L_∂z*∂z_∂v*∂v_∂w 
    ∂L_∂b = ∂L_∂z*∂z_∂v

    #init weights
    W_new = W
    b_new = b

    #update step
    W_new = W .- η * ∂L_∂w
    b_new = b .- η * sum(∂L_∂b,dims=2)

    return W_new, b_new
end

V = [-1 6 7;2 2 2;5 5 5]
V_bias = [0;0;0]
α = [5 -2]
x = [[1 3]; [4 .3]; [2 2]]
y = [-1; 20; 5;]
for i=1:10
    V, V_bias  = forwardprop(V,V_bias,x,y,α)
end
println("V:",V)
println("V_bias:",V_bias)
println("z:",FeedForward(V,V_bias,x,y,α))

{% endhighlight %}


## Query, Q
<svg width="1000" height="550">


<line x1="50" y1="50" x2="100" y2="50" style="stroke:black;stroke-width:2" />
<line x1="50" y1="80" x2="100" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="50" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="80" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="80" style="stroke:black;stroke-width:2" />




<circle cx="100" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="200" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="210" y1="50" x2="250" y2="50" style="stroke:black;stroke-width:2" />
<line x1="210" y1="80" x2="250" y2="80" style="stroke:black;stroke-width:2" />


<rect x="30" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="120" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="145" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="35" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">q1</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">Q weights</text>

<text x="400" y="45" fill="black" font-size="15">q11</text>
<polyline points="265,50 460,50 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">k11</text>
<polyline points="265,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">q11.k11</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">e1=q11.k11+q21.k21</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">q21.k21</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">q21</text>
<polyline points="265,75 360,75 360,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">k21</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,160 350,160 350,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="115" fill="black" font-size="15" font-weight="bold">k1</text>



<text x="400" y="255" fill="black" font-size="15">q11</text>
<polyline points="265,55 335,55 335,260 335,260 460,260 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="327" fill="black" font-size="15">k12</text>
<polyline points="265,330 440,330 460 260" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="260" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="272" fill="black" font-size="25">*</text>
<text x="500" y="257" fill="black" font-size="15">q11.k12</text>
<polyline points="485,260 550,260 600,317" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="317" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="325" fill="black" font-size="25">+</text>
<polyline points="630,317 740,317 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="310" fill="black" font-size="15">e2=q11.k12+q21.k22</text>
<circle cx="470" cy="380" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="377" fill="black" font-size="15">q21.k22</text>
<text x="465" y="392" fill="black" font-size="25">*</text>
<text x="400" y="377" fill="black" font-size="15">q21</text>
<polyline points="265,80 320,80 320,380 455,380 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="417" fill="black" font-size="15">k22</text>
<polyline points="485,380 550,380 600,317" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,360 280,360 280,420 440,420 460 390" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="315" fill="black" font-size="15" font-weight="bold">k2</text>

<rect x="240" y="320" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="345" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

</svg>

* Forward proprogate step:

$$\begin{aligned}
&\boldsymbol{Q}\boldsymbol{x_1}=\boldsymbol{q_1}\\
&e_1=\boldsymbol{q_1}^T\boldsymbol{k_1}\\
&e_2=\boldsymbol{q_1}^T\boldsymbol{k_2}\\
&\boldsymbol{e}=\boldsymbol{q_1}^T\boldsymbol{k}\\
&where \ \boldsymbol{Q}=\begin{bmatrix}
w_{11} \ w_{12} \ b_1 \\ w_{21} \ w_{22} \ b_2 
\end{bmatrix},
\boldsymbol{x_1}=\begin{bmatrix}
x_{11} \\ x_{21} \\ 1
\end{bmatrix},
\boldsymbol{q_1}=\begin{bmatrix}
q_{11} \\ q_{21}
\end{bmatrix},
\boldsymbol{k_1}=\begin{bmatrix}
k_{11} \\ k_{21} 
\end{bmatrix},
\boldsymbol{k_2}=\begin{bmatrix}
k_{12} \\ k_{22} 
\end{bmatrix},
\boldsymbol{k}=\begin{bmatrix}
k_{11} \ k_{12} \\ k_{21} \ k_{22}
\end{bmatrix},
\boldsymbol{e}=\begin{bmatrix}
e_1 \ e_2
\end{bmatrix}
\end{aligned}$$

Note that we have absorbed the bias terms $b_1,b_2$ into $Q$ and appended '1' to the vector $\boldsymbol{x_1}$ for compactness. 
The above steps can be written concisely in vector form,
$$\begin{aligned}
&\boldsymbol{e}=(\boldsymbol{Qx_1})^T\boldsymbol{k}\\
&\boldsymbol{e}=\boldsymbol{q_1}^T\boldsymbol{k}
\end{aligned}$$

<svg width="1000" height="200">

<rect x="275" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="275" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="360" y="95" fill="black" font-size="50" font-weight="bold">*</text>

<text x="305" y="140" fill="black" font-size="25" font-weight="bold">Q</text>
<text x="240" y="90" fill="black" font-size="50" font-weight="bold">(</text>

<rect x="390" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="390" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="390" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="390" y="140" fill="black" font-size="25" font-weight="bold">x1</text>
<text x="440" y="90" fill="black" font-size="50" font-weight="bold">)'</text>

<rect x="540" y="40" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="540" y="65" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="40" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="65" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="480" y="95" fill="black" font-size="50" font-weight="bold">*</text>
<text x="560" y="140" fill="black" font-size="25" font-weight="bold">k</text>

<text x="620" y="90" fill="black" font-size="25" font-weight="bold">=</text>
<rect x="650" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="670" y="140" fill="black" font-size="25" font-weight="bold">e</text>

</svg>

* Backpropagate step:

Let $\boldsymbol{y}$ be the 2x1 target vector.

$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{e} - \boldsymbol{y}^T||^2_2 \\
&\frac{\partial{L}}{\partial{\boldsymbol{e}}}=\boldsymbol{e}
-\boldsymbol{y}^T \\
&\frac{\partial{L}}{\partial{\boldsymbol{e}}}=\begin{bmatrix}
e_1 - y_1 \ e_2 - y_2 \\
\end{bmatrix}\\
&\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{q_1}}}=\boldsymbol{k}^T\\
&\begin{bmatrix}
\frac{\partial{e_1}}{\partial{q_{11}}} \ \frac{\partial{e_1}}{\partial{q_{21}}}  \\
\frac{\partial{e_2}}{\partial{q_{11}}} \ \frac{\partial{e_2}}{\partial{q_{21}}}  \\
\end{bmatrix} =
\begin{bmatrix}
k_{11} \ k_{21} \\
k_{12} \ k_{22} \\
\end{bmatrix}, \\ 
&\frac{\partial{\boldsymbol{q_1}}}{\partial{\boldsymbol{Q}}}=\boldsymbol{x_1}^T,\
&\frac{\partial{\boldsymbol{q_1}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1}\\
&\frac{\partial{L}}{\partial{\boldsymbol{Q}}}=\frac{\partial{L}}{\partial{\boldsymbol{e}}}\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{q_1}}}\frac{\partial{\boldsymbol{q_1}}}{\partial{\boldsymbol{Q}}}\\
&\frac{\partial{L}}{\partial{\boldsymbol{q_1}}}=\frac{\partial{L}}{\partial{\boldsymbol{e}}}\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{q_1}}}=\begin{bmatrix}
e_1 - y_1 \ e_2 - y_2 \\
\end{bmatrix}\begin{bmatrix}
k_{11} \ k_{21} \\
k_{12} \ k_{22} \\
\end{bmatrix}\\

\end{aligned}$$

<svg width="1000" height="550">


<line x1="50" y1="50" x2="100" y2="50" style="stroke:black;stroke-width:2" />
<line x1="50" y1="80" x2="100" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="50" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="80" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="80" style="stroke:black;stroke-width:2" />




<circle cx="100" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="200" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="210" y1="50" x2="250" y2="50" style="stroke:black;stroke-width:2" />
<line x1="210" y1="80" x2="250" y2="80" style="stroke:black;stroke-width:2" />


<rect x="30" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="120" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="145" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="35" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">q1</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">Q weights</text>

<text x="400" y="45" fill="black" font-size="15">q11</text>
<polyline points="265,50 460,50 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">k11</text>
<polyline points="265,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">q11.k11</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">e1=q11.k11+q21.k21</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">q21.k21</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">q21</text>
<polyline points="265,75 360,75 360,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">k21</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,160 350,160 350,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="115" fill="black" font-size="15" font-weight="bold">k1</text>



<text x="400" y="255" fill="black" font-size="15">q11</text>
<polyline points="265,55 335,55 335,260 335,260 460,260 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="327" fill="black" font-size="15">k12</text>
<polyline points="265,330 440,330 460 260" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="260" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="272" fill="black" font-size="25">*</text>
<text x="500" y="257" fill="black" font-size="15">q11.k12</text>
<polyline points="485,260 550,260 600,317" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="317" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="325" fill="black" font-size="25">+</text>
<polyline points="630,317 740,317 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="310" fill="black" font-size="15">e2=q11.k12+q21.k22</text>
<circle cx="470" cy="380" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="377" fill="black" font-size="15">q21.k22</text>
<text x="465" y="392" fill="black" font-size="25">*</text>
<text x="400" y="377" fill="black" font-size="15">q21</text>
<polyline points="265,80 320,80 320,380 455,380 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="417" fill="black" font-size="15">k22</text>
<polyline points="485,380 550,380 600,317" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,360 280,360 280,420 440,420 460 390" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="315" fill="black" font-size="15" font-weight="bold">k2</text>

<rect x="240" y="320" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="345" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<polyline points="680,120 670,115 680 110" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="680,320 670,315 680 310" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,65 490,60 500 55" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,185 490,180 500 175" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,265 490,260 500 255" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,385 490,380 500 375" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,55 370,50 380 45" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,185 370,180 380 175" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,265 370,260 380 255" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,385 370,380 380 375" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="225,55 215,50 225 45" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="225,85 215,80 225 75" style="fill:none;stroke:red;stroke-width:3" />

<text x="680" y="130" fill="red" font-size="15">∂L/∂e1</text>
<text x="680" y="330" fill="red" font-size="15">∂L/∂e2</text>
<text x="500" y="75" fill="red" font-size="15">∂L/∂e1</text>
<text x="500" y="195" fill="red" font-size="15">∂L/∂e1</text>
<text x="500" y="275" fill="red" font-size="15">∂L/∂e2</text>
<text x="500" y="395" fill="red" font-size="15">∂L/∂e2</text>
<text x="380" y="65" fill="red" font-size="15">k11∂L/∂e1</text>
<text x="380" y="195" fill="red" font-size="15">k21∂L/∂e1</text>
<text x="380" y="275" fill="red" font-size="15">k12∂L/∂e2</text>
<text x="380" y="395" fill="red" font-size="15">k22∂L/∂e2</text>
<text x="100" y="15" fill="red" font-size="15">k11∂L/∂e1+k12∂L/∂e2</text>
<text x="100" y="105" fill="red" font-size="15">k21∂L/∂e1+k22∂L/∂e2</text>
</svg>




* Update step:

$$\begin{aligned}
&\boldsymbol{Q}^{new}=\boldsymbol{Q}-\eta\frac{\partial{L}}{\partial{\boldsymbol{Q}}}\\
&\boldsymbol{Q}^{new}=\boldsymbol{Q}-\eta(\boldsymbol{e}-\boldsymbol{y}^T)\boldsymbol{k}^T\boldsymbol{x_1}^T\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="200" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="200" y="140" fill="black" font-size="25" font-weight="bold">Qnew</text>
<text x="270" y="70" fill="black" font-size="25">=</text>
<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="315" y="140" fill="black" font-size="25" font-weight="bold">Q</text>
<text x="360" y="70" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>




<text x="440" y="140" fill="black" font-size="25" font-weight="bold">(e-y')k'</text>

<rect x="540" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="565" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="510" y="70" fill="black" font-size="25" font-weight="bold">*</text>
<text x="560" y="140" fill="black" font-size="25" font-weight="bold">x'</text>

<text x="620" y="70" fill="black" font-size="25" font-weight="bold">=</text>
<rect x="650" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>



<text x="650" y="140" fill="black" font-size="25" font-weight="bold">(e-y')k'x'</text>
<text x="750" y="70" fill="black" font-size="25" font-weight="bold">)</text>
</svg>


$$\begin{aligned}
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta\frac{\partial{L}}{\partial{\boldsymbol{b}}}\\
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta \ ((\boldsymbol{e}-\boldsymbol{y}^T)\boldsymbol{k}^T)^T\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="200" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="200" y="140" fill="black" font-size="25" font-weight="bold">bnew</text>
<text x="270" y="70" fill="black" font-size="25">=</text>
<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="315" y="140" fill="black" font-size="25" font-weight="bold">b</text>
<text x="360" y="70" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="440" y="140" fill="black" font-size="25" font-weight="bold">((e-y')k')'</text>

<text x="550" y="70" fill="black" font-size="25" font-weight="bold">)'</text>
</svg>

The following julia code illustrates an example with 3x1 ks: $\boldsymbol{k_1},\boldsymbol{k_2},\boldsymbol{k_3}$, 3x4 $Q$ matrix and a 4x1 $\boldsymbol{x}$ input vector.

{% highlight julia%}
# Squared Loss
function Squared_Loss(z,x)
    return 0.5*(L2norm(z-x))^2
end

# L2 norm
function L2norm(x)
    sqrt(sum(x.^2))
end

#Feed forward
function FeedForward(Q,k,x,y)
    q = Q*x
    e = q'*k
    return q,e
end

# Forward propagate
function forwardprop(Q,k,x,y)
    q, e = FeedForward(Q,k,x,y)
    return backprop(Q,k,x,q,e,y)
end

# Backpropagate
function backprop(Q,k,x,q,e,y,η=.02)
    println(Squared_Loss(e',y))

    ∂L_∂e = (e-y')
    ∂e_∂q = k'
    ∂q_∂Q = x[1:end-1]'

    ∂L_∂q = ∂L_∂e*∂e_∂q
    ∂L_∂Q = ∂L_∂q.*∂q_∂Q
    ∂L_∂Qb = ∂L_∂q

    #init Q weights
    Q_new = Q

    #update step
    Q_new[:,1:end-1] = Q_new[:,1:end-1] .- η * ∂L_∂Q
    Q_new[:,end:end] = Q_new[:,end:end] .- η * ∂L_∂Qb' 

    return Q_new
end


Q = [1. 5 .2 0; 1 2 .4 0; 4 5 1 0;] #Last column is bias terms initialized to 0
k = [1 -3 6; 2 0 1; 4 5 1;]
x = [0.5; 0.5; .3; 1] #Last row is bias term set to 1
y = [-3.14; -6.3; 2.21;]

for i=1:100
    Q = forwardprop(Q,k,x,y)
end
println("Q:",Q)
println("e:",FeedForward(Q,k,x,y)[2])
{% endhighlight %}

## Keys, K
<svg width="1000" height="550">


<line x1="50" y1="50" x2="100" y2="50" style="stroke:black;stroke-width:2" />
<line x1="50" y1="80" x2="100" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="50" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="80" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<circle cx="100" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="200" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="210" y1="50" x2="250" y2="50" style="stroke:black;stroke-width:2" />
<line x1="210" y1="80" x2="250" y2="80" style="stroke:black;stroke-width:2" />


<rect x="30" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="120" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="145" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="35" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">q1</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">Q weights</text>

<line x1="50" y1="250" x2="100" y2="250" style="stroke:black;stroke-width:2" />
<line x1="50" y1="280" x2="100" y2="280" style="stroke:black;stroke-width:2" />

<line x1="110" y1="250" x2="170" y2="250" style="stroke:black;stroke-width:2" />
<line x1="110" y1="250" x2="170" y2="280" style="stroke:black;stroke-width:2" />

<line x1="110" y1="280" x2="170" y2="250" style="stroke:black;stroke-width:2" />
<line x1="110" y1="280" x2="170" y2="280" style="stroke:black;stroke-width:2" />




<circle cx="100" cy="250" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="280" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="180" cy="250" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="180" cy="280" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="190" y1="250" x2="210" y2="250" style="stroke:black;stroke-width:2" />
<line x1="190" y1="280" x2="210" y2="280" style="stroke:black;stroke-width:2" />


<rect x="30" y="240" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="265" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="55" y="240" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="55" y="265" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="200" y="240" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="265" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="240" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="265" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="35" y="230" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="60" y="230" fill="black" font-size="15" font-weight="bold">x2</text>

<text x="205" y="230" fill="black" font-size="15" font-weight="bold">k1</text>
<text x="230" y="230" fill="black" font-size="15" font-weight="bold">k2</text>
<text x="110" y="230" fill="black" font-size="15" font-weight="bold">K weights</text>
<line x1="210" y1="217" x2="240" y2="170" style="stroke:red;stroke-width:4" />
<line x1="235" y1="290" x2="240" y2="320" style="stroke:red;stroke-width:4" />
<rect x="190" y="215" width="35" height="75" style="fill:none;stroke-width:2;stroke:rgb(200,0,0)"></rect>
<rect x="225" y="215" width="35" height="75" style="fill:none;stroke-width:2;stroke:rgb(200,0,0)"></rect>

<text x="400" y="45" fill="black" font-size="15">q11</text>
<polyline points="265,50 460,50 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">k11</text>
<polyline points="265,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">q11.k11</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">e1=q11.k11+q21.k21</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">q21.k21</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">q21</text>
<polyline points="265,75 360,75 360,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">k21</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,160 350,160 350,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="115" fill="black" font-size="15" font-weight="bold">k1</text>



<text x="400" y="255" fill="black" font-size="15">q11</text>
<polyline points="265,55 335,55 335,260 335,260 460,260 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="327" fill="black" font-size="15">k12</text>
<polyline points="265,330 440,330 460 260" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="260" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="272" fill="black" font-size="25">*</text>
<text x="500" y="257" fill="black" font-size="15">q11.k12</text>
<polyline points="485,260 550,260 600,317" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="317" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="325" fill="black" font-size="25">+</text>
<polyline points="630,317 740,317 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="310" fill="black" font-size="15">e2=q11.k12+q21.k22</text>
<circle cx="470" cy="380" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="377" fill="black" font-size="15">q21.k22</text>
<text x="465" y="392" fill="black" font-size="25">*</text>
<text x="400" y="377" fill="black" font-size="15">q21</text>
<polyline points="265,80 320,80 320,380 455,380 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="417" fill="black" font-size="15">k22</text>
<polyline points="485,380 550,380 600,317" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,360 280,360 280,420 440,420 460 390" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="315" fill="black" font-size="15" font-weight="bold">k2</text>

<rect x="240" y="320" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="345" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

</svg>

* Forward proprogate step:

$$\begin{aligned}
&\boldsymbol{K}\boldsymbol{x}=\boldsymbol{k}\\
&e_1=\boldsymbol{q_1}^T\boldsymbol{k_1}\\
&e_2=\boldsymbol{q_1}^T\boldsymbol{k_2}\\
&\boldsymbol{e}=\boldsymbol{q_1}^T\boldsymbol{k}\\
&where \ \boldsymbol{K}=\begin{bmatrix}
w_{11} \ w_{12} \ b_1 \\ w_{21} \ w_{22} \ b_2 
\end{bmatrix},
\boldsymbol{x}=\begin{bmatrix} \boldsymbol{x_1} \ \boldsymbol{x_2} \end{bmatrix}=
\begin{bmatrix}
x_{11} \ x_{12} \\ x_{21} \ x_{22}  \\ 1 \ \ 1 \\

\end{bmatrix},
\boldsymbol{q_1}=\begin{bmatrix}
q_{11} \\ q_{21}
\end{bmatrix},
\boldsymbol{k_1}=\begin{bmatrix}
k_{11} \\ k_{21} 
\end{bmatrix},
\boldsymbol{k_2}=\begin{bmatrix}
k_{12} \\ k_{22} 
\end{bmatrix},
\boldsymbol{k}=\begin{bmatrix}
k_{11} \ k_{12} \\ k_{21} \ k_{22}
\end{bmatrix},
\boldsymbol{e}=\begin{bmatrix}
e_1 \ e_2
\end{bmatrix}
\end{aligned}$$

Note that as before we have absorbed the bias terms $b_1,b_2$ into $K$ and appended '1's to the vectors $\boldsymbol{x_1}$, $\boldsymbol{x_2}$ for compactness. 
The above steps can be written concisely in vector form,
$$\begin{aligned}
&\boldsymbol{e}=(\boldsymbol{Qx_1})^T\boldsymbol{k}\\
&\boldsymbol{e}=\boldsymbol{q_1}^T\boldsymbol{k}
\end{aligned}$$

* Backpropagate step:

Let $\boldsymbol{y}$ be the 2x1 target vector.

$$\begin{aligned}
&L = \frac{1}{2}||\boldsymbol{e} - \boldsymbol{y}^T||^2_2 \\
&\frac{\partial{L}}{\partial{\boldsymbol{e}}}=\boldsymbol{e}
-\boldsymbol{y}^T \\
&\frac{\partial{L}}{\partial{\boldsymbol{e}}}=\begin{bmatrix}
e_1 - y_1 \ e_2 - y_2 \\
\end{bmatrix}\\
&\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{k}}}=\boldsymbol{q_1}^T\\
&\begin{bmatrix}
\frac{\partial{e_1}}{\partial{k_{11}}} \ \frac{\partial{e_1}}{\partial{k_{21}}}  \\
\frac{\partial{e_2}}{\partial{k_{12}}} \ \frac{\partial{e_2}}{\partial{k_{22}}}  \\
\end{bmatrix} =
\begin{bmatrix}
q_{11} \ q_{21} \\
q_{11} \ q_{21} \\
\end{bmatrix}, \\ 
&\frac{\partial{\boldsymbol{k}}}{\partial{\boldsymbol{K}}}=\boldsymbol{x}^T,\
&\frac{\partial{\boldsymbol{k}}}{\partial{\boldsymbol{b}}}=\boldsymbol{1}\\
&\frac{\partial{L}}{\partial{\boldsymbol{K}}}=\frac{\partial{L}}{\partial{\boldsymbol{e}}}\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{k}}}\frac{\partial{\boldsymbol{k}}}{\partial{\boldsymbol{K}}}\\
&\frac{\partial{L}}{\partial{\boldsymbol{k}}}=\frac{\partial{L}}{\partial{\boldsymbol{e}}}\frac{\partial{\boldsymbol{e}}}{\partial{\boldsymbol{k}}}=\begin{bmatrix}
e_1 - y_1 \\ e_2 - y_2 \\
\end{bmatrix}\begin{bmatrix}
q_{11} \ q_{21} \\
\end{bmatrix}\\
&\frac{\partial{L}}{\partial{\boldsymbol{k}}}=
\begin{bmatrix}
\frac{\partial{L}}{\partial{\boldsymbol{k_1}}}\\ \frac{\partial{L}}{\partial{\boldsymbol{k_2}}} \\
\end{bmatrix}=

\begin{bmatrix}
q_{11} (e_1 - y_1) \ \ q_{21} (e_1 - y_1) \\ q_{11}(e_2 - y_2) \ \ q_{21} (e_2 - y_2) \\
\end{bmatrix}
\end{aligned}$$



<svg width="1000" height="550">

<line x1="50" y1="50" x2="100" y2="50" style="stroke:black;stroke-width:2" />
<line x1="50" y1="80" x2="100" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="50" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="50" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<line x1="110" y1="80" x2="190" y2="50" style="stroke:black;stroke-width:2" />
<line x1="110" y1="80" x2="190" y2="80" style="stroke:black;stroke-width:2" />

<circle cx="100" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="200" cy="50" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="200" cy="80" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="210" y1="50" x2="250" y2="50" style="stroke:black;stroke-width:2" />
<line x1="210" y1="80" x2="250" y2="80" style="stroke:black;stroke-width:2" />


<rect x="30" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="90" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="240" y="40" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="65" width="25" height="25" style="fill:rgb(200,0,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="120" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="145" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>


<text x="35" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">q1</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">Q weights</text>

<line x1="50" y1="250" x2="100" y2="250" style="stroke:black;stroke-width:2" />
<line x1="50" y1="280" x2="100" y2="280" style="stroke:black;stroke-width:2" />

<line x1="110" y1="250" x2="170" y2="250" style="stroke:black;stroke-width:2" />
<line x1="110" y1="250" x2="170" y2="280" style="stroke:black;stroke-width:2" />

<line x1="110" y1="280" x2="170" y2="250" style="stroke:black;stroke-width:2" />
<line x1="110" y1="280" x2="170" y2="280" style="stroke:black;stroke-width:2" />




<circle cx="100" cy="250" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="100" cy="280" r="10" stroke="black" stroke-width="1" fill="white"></circle>

<circle cx="180" cy="250" r="10" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="180" cy="280" r="10" stroke="black" stroke-width="1" fill="white"></circle>



<line x1="190" y1="250" x2="210" y2="250" style="stroke:black;stroke-width:2" />
<line x1="190" y1="280" x2="210" y2="280" style="stroke:black;stroke-width:2" />


<rect x="30" y="240" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="30" y="265" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="55" y="240" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="55" y="265" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<rect x="200" y="240" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="265" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="240" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="265" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="35" y="230" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="60" y="230" fill="black" font-size="15" font-weight="bold">x2</text>

<text x="205" y="230" fill="black" font-size="15" font-weight="bold">k1</text>
<text x="230" y="230" fill="black" font-size="15" font-weight="bold">k2</text>
<text x="110" y="230" fill="black" font-size="15" font-weight="bold">K weights</text>
<line x1="210" y1="217" x2="240" y2="170" style="stroke:red;stroke-width:4" />
<line x1="235" y1="290" x2="240" y2="320" style="stroke:red;stroke-width:4" />
<rect x="190" y="215" width="35" height="75" style="fill:none;stroke-width:2;stroke:rgb(200,0,0)"></rect>
<rect x="225" y="215" width="35" height="75" style="fill:none;stroke-width:2;stroke:rgb(200,0,0)"></rect>

<text x="400" y="45" fill="black" font-size="15">q11</text>
<polyline points="265,50 460,50 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">k11</text>
<polyline points="265,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">q11.k11</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">e1=q11.k11+q21.k21</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">q21.k21</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">q21</text>
<polyline points="265,75 360,75 360,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">k21</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,160 350,160 350,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="115" fill="black" font-size="15" font-weight="bold">k1</text>



<text x="400" y="255" fill="black" font-size="15">q11</text>
<polyline points="265,55 335,55 335,260 335,260 460,260 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="327" fill="black" font-size="15">k12</text>
<polyline points="265,330 440,330 460 260" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="260" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="272" fill="black" font-size="25">*</text>
<text x="500" y="257" fill="black" font-size="15">q11.k12</text>
<polyline points="485,260 550,260 600,317" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="317" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="325" fill="black" font-size="25">+</text>
<polyline points="630,317 740,317 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="310" fill="black" font-size="15">e2=q11.k12+q21.k22</text>
<circle cx="470" cy="380" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="377" fill="black" font-size="15">q21.k22</text>
<text x="465" y="392" fill="black" font-size="25">*</text>
<text x="400" y="377" fill="black" font-size="15">q21</text>
<polyline points="265,80 320,80 320,380 455,380 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="417" fill="black" font-size="15">k22</text>
<polyline points="485,380 550,380 600,317" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,360 280,360 280,420 440,420 460 390" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="315" fill="black" font-size="15" font-weight="bold">k2</text>

<rect x="240" y="320" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="345" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="35" y="30" fill="black" font-size="15" font-weight="bold">x1</text>
<text x="245" y="30" fill="black" font-size="15" font-weight="bold">q1</text>
<text x="110" y="30" fill="black" font-size="15" font-weight="bold">Q weights</text>

<text x="400" y="45" fill="black" font-size="15">q11</text>
<polyline points="265,50 460,50 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="127" fill="black" font-size="15">k11</text>
<polyline points="265,130 440,130 460 60" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="72" fill="black" font-size="25">*</text>
<text x="500" y="57" fill="black" font-size="15">q11.k11</text>
<polyline points="485,60 550,60 600,117" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="117" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="125" fill="black" font-size="25">+</text>
<polyline points="630,117 740,117 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="110" fill="black" font-size="15">e1=q11.k11+q21.k21</text>
<circle cx="470" cy="180" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="177" fill="black" font-size="15">q21.k21</text>
<text x="465" y="192" fill="black" font-size="25">*</text>
<text x="400" y="177" fill="black" font-size="15">q21</text>
<polyline points="265,75 360,75 360,180 455,180 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="217" fill="black" font-size="15">k21</text>
<polyline points="485,180 550,180 600,117" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,160 350,160 350,220 440,220 460 190" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="115" fill="black" font-size="15" font-weight="bold">k1</text>



<text x="400" y="255" fill="black" font-size="15">q11</text>
<polyline points="265,55 335,55 335,260 335,260 460,260 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="327" fill="black" font-size="15">k12</text>
<polyline points="265,330 440,330 460 260" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="470" cy="260" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="465" y="272" fill="black" font-size="25">*</text>
<text x="500" y="257" fill="black" font-size="15">q11.k12</text>
<polyline points="485,260 550,260 600,317" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="615" cy="317" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="607" y="325" fill="black" font-size="25">+</text>
<polyline points="630,317 740,317 " style="fill:none;stroke:black;stroke-width:1" />
<text x="640" y="310" fill="black" font-size="15">e2=q11.k12+q21.k22</text>
<circle cx="470" cy="380" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="500" y="377" fill="black" font-size="15">q21.k22</text>
<text x="465" y="392" fill="black" font-size="25">*</text>
<text x="400" y="377" fill="black" font-size="15">q21</text>
<polyline points="265,80 320,80 320,380 455,380 " style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="417" fill="black" font-size="15">k22</text>
<polyline points="485,380 550,380 600,317" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="265,360 280,360 280,420 440,420 460 390" style="fill:none;stroke:black;stroke-width:1" />
<text x="245" y="315" fill="black" font-size="15" font-weight="bold">k2</text>

<rect x="240" y="320" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="240" y="345" width="25" height="25" style="fill:rgb(30,100,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<polyline points="680,120 670,115 680 110" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="680,320 670,315 680 310" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,65 490,60 500 55" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,185 490,180 500 175" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,265 490,260 500 255" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="500,385 490,380 500 375" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,135 370,130 380 125" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,225 370,220 380 215" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,335 370,330 380 325" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="380,425 370,420 380 415" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="200,255 190,250 200,245" style="fill:none;stroke:red;stroke-width:3" />
<polyline points="200,285 190,280 200,275" style="fill:none;stroke:red;stroke-width:3" />

<text x="680" y="130" fill="red" font-size="15">∂L/∂e1</text>
<text x="680" y="330" fill="red" font-size="15">∂L/∂e2</text>
<text x="500" y="75" fill="red" font-size="15">∂L/∂e1</text>
<text x="500" y="195" fill="red" font-size="15">∂L/∂e1</text>
<text x="500" y="275" fill="red" font-size="15">∂L/∂e2</text>
<text x="500" y="395" fill="red" font-size="15">∂L/∂e2</text>
<text x="380" y="145" fill="red" font-size="15">q11∂L/∂e1</text>
<text x="380" y="235" fill="red" font-size="15">q21∂L/∂e1</text>
<text x="380" y="345" fill="red" font-size="15">q11∂L/∂e2</text>
<text x="380" y="435" fill="red" font-size="15">q21∂L/∂e2</text>
<text x="70" y="195" fill="red" font-size="15">[q11∂L/∂e1;q21∂L/∂e1]</text>
<text x="80" y="315" fill="red" font-size="15">[q11∂L/∂e2;q21∂L/∂e2]</text>
</svg>

* Update step:

$$\begin{aligned}
&\boldsymbol{K}^{new}=\boldsymbol{K}-\eta\frac{\partial{L}}{\partial{\boldsymbol{K}}}\\
&\boldsymbol{K}^{new}=\boldsymbol{K}-\eta((\boldsymbol{e}-\boldsymbol{y}^T)^T\boldsymbol{q_1}^T)^T\boldsymbol{x^T}\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="200" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="225" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="200" y="140" fill="black" font-size="25" font-weight="bold">Knew</text>
<text x="270" y="70" fill="black" font-size="25">=</text>
<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="40" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="325" y="65" width="25" height="25" style="fill:rgb(0,100,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="315" y="140" fill="black" font-size="25" font-weight="bold">K</text>
<text x="360" y="70" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>



<text x="440" y="140" fill="black" font-size="25" font-weight="bold">q1(e-y')</text>

<rect x="540" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="40" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="540" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="565" y="65" width="25" height="25" style="fill:rgb(160,100,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="510" y="70" fill="black" font-size="25" font-weight="bold">*</text>
<text x="560" y="140" fill="black" font-size="25" font-weight="bold">x'</text>

<text x="620" y="70" fill="black" font-size="25" font-weight="bold">=</text>
<rect x="650" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="40" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="650" y="65" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="675" y="65" width="25" height="25" style="fill:rgb(100,160,100);stroke-width:3;stroke:rgb(0,0,0)"></rect>


<text x="650" y="140" fill="black" font-size="25" font-weight="bold">q1(e-y')x'</text>
<text x="750" y="70" fill="black" font-size="25" font-weight="bold">)</text>
</svg>


$$\begin{aligned}
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta\frac{\partial{L}}{\partial{\boldsymbol{b}}}\\
&\boldsymbol{b}^{new}=\boldsymbol{b}-\eta \ sumcols((\boldsymbol{e}-\boldsymbol{y}^T)^T\boldsymbol{q_1}^T)^T\\
\end{aligned}
$$

<svg width="1000" height="200">

<rect x="200" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="200" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="200" y="140" fill="black" font-size="25" font-weight="bold">bnew</text>
<text x="270" y="70" fill="black" font-size="25">=</text>
<rect x="300" y="40" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="300" y="65" width="25" height="25" style="fill:rgb(0,200,0);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="315" y="140" fill="black" font-size="25" font-weight="bold">b</text>
<text x="360" y="70" fill="black" font-size="25" font-weight="bold">-η (</text>

<rect x="450" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="450" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="475" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="380" y="140" fill="black" font-size="25" font-weight="bold">sumcols(q1(e-y'))</text>
<polyline points="510,35 400,35 430,25 400,35 430,45 " style="fill:none;stroke:black;stroke-width:5" />
<text x="430" y="20" fill="black" font-size="25" font-weight="bold">sum cols</text>

<rect x="550" y="40" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="550" y="65" width="25" height="25" style="fill:rgb(100,0,160);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="510" y="70" fill="black" font-size="25" font-weight="bold">=</text>
<text x="600" y="70" fill="black" font-size="25" font-weight="bold">)</text>
</svg>

The following julia code illustrates an example with 3x1 ks: $\boldsymbol{k_1},\boldsymbol{k_2},\boldsymbol{k_3}$, 3x4 $K$ matrix and a 4x3 $\boldsymbol{x}$ input vector.

{% highlight julia %}

# Squared Loss
function Squared_Loss(z,x)
    return 0.5*(L2norm(z-x))^2
end

# L2 norm
function L2norm(x)
    sqrt(sum(x.^2))
end

#Feed forward
function FeedForward(K,q,x,y)
    k = K*x
    e = q'*k
    return k,e
end

# Forward propagate
function forwardprop(K,q,x,y)
    k, e = FeedForward(K,q,x,y)
    return backprop(K,k,x,q,e,y)
end

# Backpropagate
function backprop(K,k,x,q,e,y,η=.01)
    println(Squared_Loss(e',y))
  
    ∂L_∂e = (e-y')'
    ∂e_∂k = q'
    ∂k_∂K = x[1:end-1,:]'

    ∂L_∂k = ∂L_∂e*∂e_∂k
    ∂L_∂K = ∂L_∂k'*∂k_∂K
    ∂L_∂Kb = ∂L_∂k'

    #init K weights
    K_new = K

    #update step
    K_new[:,1:end-1] = K_new[:,1:end-1] .- η * ∂L_∂K
    K_new[:,end:end] = K_new[:,end:end] .- η * sum(∂L_∂Kb,dims=2) 

    return K_new
end


K = [1 -3 6. 0; 2 0 1 0; 4 5 1 0;] #Last column is bias terms initialized to 0
q = [1.; 5; .2;]
x = [0.5 .2 .4; 0.5 .2 .6; .3 .25 .7; 1 1 1;] #Last row are bias terms set to 1
y = [-3.14; -6.3; 2.21;]

for i=1:300
    K = forwardprop(K,q,x,y)
end
println("K:",K)
println("e:",FeedForward(K,q,x,y)[2])
{% endhighlight %}

## Softmax
<svg width="1000" height="200">

<rect x="60" y="20" width="75" height="100" style="fill:none;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="15" y="45" fill="black" font-size="15">e1</text>
<text x="15" y="75" fill="black" font-size="15">e2</text>

<text x="70" y="75" fill="black" font-size="15">softmax</text>
<polyline points="5,50 60,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="5,80 60,80" style="fill:none;stroke:black;stroke-width:1" />

<polyline points="135,50 200,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="135,80 200,80" style="fill:none;stroke:black;stroke-width:1" />

<text x="155" y="45" fill="black" font-size="15">α1</text>
<text x="155" y="75" fill="black" font-size="15">α2</text>

<polyline points="295,20 370,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="295,100 370,100" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="370" cy="20" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="370" cy="100" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<text x="360" y="20" fill="black" font-size="15">exp</text>
<text x="360" y="100" fill="black" font-size="15">exp</text>

<text x="330" y="25" fill="black" font-size="20">></text>
<text x="330" y="105" fill="black" font-size="20">></text>
<text x="580" y="25" fill="black" font-size="20">></text>
<text x="580" y="105" fill="black" font-size="20">></text>

<polyline points="385,20 550,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="385,100 550,100" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="450" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="550" cy="20" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="550" cy="100" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="550" y="25" fill="black" font-size="20">/</text>
<text x="550" y="105" fill="black" font-size="20">/</text>
<text x="445" y="65" fill="black" font-size="20">+</text>

<polyline points="385,20 420,50, 440,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="385,100 420,70, 440,70" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="465,60 500,60, 540,30" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="465,60 500,60, 540,90" style="fill:none;stroke:black;stroke-width:1" />

<polyline points="565,20 620,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="565,100 620,100" style="fill:none;stroke:black;stroke-width:1" />

<text x="450" y="15" fill="black" font-size="15">a</text>
<text x="450" y="95" fill="black" font-size="15">c</text>
<text x="490" y="55" fill="black" font-size="15">b</text>
<text x="305" y="15" fill="black" font-size="15">e1</text>
<text x="305" y="95" fill="black" font-size="15">e2</text>
<text x="600" y="15" fill="black" font-size="15">α1=exp(e1)/(exp(e1)+exp(e2))</text>
<text x="600" y="95" fill="black" font-size="15">α2=exp(e2)/(exp(e1)+exp(e2))</text>


</svg>

The left diagram above shows an abstraction of the softmax function (blackbox). If we pass in 2 values $e_1,e_2$ through the 'blackbox' softmax function we get 2 values $\alpha_1, \alpha_2$ out. The right diagram shows the circuit-like representation (internals) of the softmax function (blackbox).
* Forward step
$$\begin{aligned}
softmax \ function, \sigma{(x_i)}=\frac{e^{x_i}}{\sum_{i=1}^2 {e^{x_i}}} \\
for \ i=1,2
\end{aligned}$$

* Backpropagate step

$$\begin{aligned}
\frac{\partial{L}}{\partial{\boldsymbol{\alpha}}}&=
\begin{bmatrix}
\frac{\partial{L}}{\partial{\alpha_1}} \ \frac{\partial{L}}{\partial{\alpha_2}} 
\end{bmatrix}\\
\frac{\partial{L}}{\partial{e_i}}&=
\frac{\partial{L}}{\partial{\boldsymbol{\alpha}}}\frac{\partial{\boldsymbol{\alpha}}}{\partial{e_i}}\\
\frac{\partial{\boldsymbol{\alpha}}}{\partial{e_i}}&=
\begin{bmatrix}
\frac{\partial{\alpha_1}}{\partial{e_i}}  \\ \frac{\partial{\alpha_2}}{\partial{e_i}} 
\end{bmatrix}\\
\frac{\partial{L}}{\partial{e_1}}&=
\frac{\partial{L}}{\partial{\alpha_1}}\frac{\partial{\alpha_1}}{\partial{e_1}}+\frac{\partial{L}}{\partial{\alpha_2}}\frac{\partial{\alpha_2}}{\partial{e_1}}\\
&=\frac{\partial{L}}{\partial{\alpha_1}}(\alpha_1)(1-\alpha_1)+\frac{\partial{L}}{\partial{\alpha_2}}(-\alpha_1\alpha_2)\\
\frac{\partial{L}}{\partial{e_2}}&=\frac{\partial{L}}{\partial{\alpha_1}}\frac{\partial{\alpha_1}}{\partial{e_2}}+\frac{\partial{L}}{\partial{\alpha_2}}\frac{\partial{\alpha_2}}{\partial{e_2}}\\
&=\frac{\partial{L}}{\partial{\alpha_1}}(-\alpha_1\alpha_2)+\frac{\partial{L}}{\partial{\alpha_1}}(\alpha_2)(1-\alpha_2)\\
\end{aligned}$$
<svg width="1000" height="200">



<polyline points="295,20 370,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="295,100 370,100" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="370" cy="20" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="370" cy="100" r="15" stroke="black" stroke-width="1" fill="white"></circle>

<text x="360" y="20" fill="black" font-size="15">exp</text>
<text x="360" y="100" fill="black" font-size="15">exp</text>


<polyline points="385,20 550,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="385,100 550,100" style="fill:none;stroke:black;stroke-width:1" />
<circle cx="450" cy="60" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="550" cy="20" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="550" cy="100" r="15" stroke="black" stroke-width="1" fill="white"></circle>
<text x="550" y="25" fill="black" font-size="20">/</text>
<text x="550" y="105" fill="black" font-size="20">/</text>
<text x="445" y="65" fill="black" font-size="20">+</text>

<polyline points="385,20 420,50, 440,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="385,100 420,70, 440,70" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="465,60 500,60, 540,30" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="465,60 500,60, 540,90" style="fill:none;stroke:black;stroke-width:1" />

<polyline points="565,20 620,20" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="565,100 620,100" style="fill:none;stroke:black;stroke-width:1" />

<text x="470" y="27" fill="red" font-size="25"><</text>
<text x="510" y="75" fill="blue" font-size="25" rotate="45"><</text>
<text x="393" y="35" fill="red" font-size="25" rotate="45"><</text>
<text x="399" y="42" fill="blue" font-size="25" rotate="45"><</text>
<text x="520" y="55" fill="red" font-size="25" rotate="-45"><</text>

<text x="420" y="12" fill="red" font-size="15">∂α1/∂a</text>
<text x="540" y="55" fill="red" font-size="15">∂α1/∂b</text>
<text x="350" y="55" fill="red" font-size="15">∂α1/∂b</text>
<text x="540" y="75" fill="blue" font-size="15">∂α2/∂b</text>
<text x="290" y="55" fill="blue" font-size="15">∂α2/∂b</text>

<text x="305" y="15" fill="black" font-size="15">∂L/∂e1</text>
<text x="305" y="95" fill="black" font-size="15">∂L/∂e2</text>
<text x="570" y="15" fill="red" font-size="15">∂L/∂α1</text>
<text x="570" y="95" fill="blue" font-size="15">∂L/∂α2</text>
</svg>

To see this visually as signals flowing back, let $a=e^{e_1}, c=e^{e_1}, b=e^{e_1}+e^{e_2}, \alpha_1=\frac{a}{b},\alpha_2=\frac{c}{b}$. Next, let us rewrite the division operator as a product of $a$,$1/b$.\\
Refering to the above figure, we have red and blue arrows denoting signals from $\alpha_1$ and $\alpha_2$ respectively. Let us first zoom in on the $\alpha_1$ output. On the backward pass, the signal forks into 2 branches - top and mid branch.
On the top branch,
$$\begin{aligned}
\frac{\partial{\alpha_1}}{\partial{a}}&=\frac{1}{b}\\
\frac{\partial{a}}{\partial{e_1}}&=e^{e_1}\\
(\frac{\partial{\alpha_1}}{\partial{e_1}})_{top}&=\frac{e^{e_1}}{b}\\
\end{aligned}$$
On the mid branch,
$$\begin{aligned}
\frac{\partial{\alpha_1}}{\partial{b}}&=-\frac{a}{b^2}\\
\frac{\partial{b}}{\partial{e_1}}&=e^{e_1}\\
(\frac{\partial{\alpha_1}}{\partial{e_1}})_{mid}&=-\frac{e^{e_1}a}{b^2}\\
\end{aligned}$$
As both paths converge at the 'exp' node we add them together giving
$$\begin{aligned}
\frac{\partial{\alpha_1}}{\partial{e_1}}&=\frac{e^{e_1}}{b}-\frac{e^{e_1}a}{b^2}\\
&=\frac{e^{e_1}}{b}(1-\frac{a}{b})\\
&=\alpha_1(1-\alpha_1)
\end{aligned}$$

Now zooming in to the $\alpha_2$ output and working backwards, we see that we have a path to $e_1$ via the $b$ mid branch.
On the mid branch,
$$\begin{aligned}
\frac{\partial{\alpha_2}}{\partial{b}}&=-\frac{c}{b^2}\\
\frac{\partial{b}}{\partial{e_1}}&=e^{e_1}\\
(\frac{\partial{\alpha_2}}{\partial{e_1}})_{mid}&=-\frac{e^{e_1}c}{b^2}\\
&=-\alpha_1\alpha_2
\end{aligned}$$

The same steps can be applied to obtain the gradients for $e_2$.

## References
1. [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)
2. [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/)
3. [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf)