---
layout: sidebar
title:  "Attention from Scratch in Julia"
date:   2024-01-28 21:16:00 +0100
categories: ML 
---

* toc
{:toc}

## Overview
This post explores the attention mechanism by building it up from scratch and applying it to a crude and rudimentary sentiment analysis example.


## Embeddings
We start with a sentence. A sentence with $n$ words is essentially a sequence of words $w_1,w_2...w_n$ .
To process them numerically, we need to map words into a vector representation i.e. $word\ w_i \to vector\ x_i$
To do so, we can use pretrained word embeddings out there such as [GloVe](https://nlp.stanford.edu/pubs/glove.pdf). This is represented by 

$$\begin{equation}E(w_i) = x_i \tag{1}\label{eq:embeddings}\end{equation}$$ 
where $E$ is an embeddings transformation.

First import an embeddings package and load in GloVe embeddings. 
{% highlight julia  %}
using Embeddings
const embtable = load_embeddings(GloVe{:en},1,max_vocab_size=10000)
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))
{% endhighlight %}

Next we write 2 helper functions -  <b><i>get_embeddings</i></b> that looks up a word in <b><i>embtable</i></b> and returns the embeddings (\ref{eq:embeddings}) and <b><i>word_tokeniser</i></b> that splits up a sentence into a vector of words. 
{% highlight julia  %}
#Returns embeddings for word
function get_embeddings(word)
    return embtable.embeddings[:,get_word_index[word]]
end

#Splits sentence into a vector of words
function word_tokeniser(sentence)
    return split(sentence," ")
end

{% endhighlight %}
We can try it out in the julia REPL.
{% highlight julia  %}
julia> get_embeddings("red")
50-element Vector{Float32}:
 -0.12878
  0.8798
 -0.60694
  0.12934
  0.5868
 -0.038246
 -1.0408
 -0.52881
 -0.29563
 -0.72567
  0.21189
  0.17112
  0.19173
  ⋮
  0.050825
 -0.20362
  0.13695
  0.26686
 -0.19461
 -0.75482
  1.0303
 -0.057467
 -0.32327
 -0.7712
 -0.16764
 -0.73835

julia> word_tokeniser("This is a sentence")
4-element Vector{SubString{String}}:
 "This"
 "is"
 "a"
 "sentence"
 {% endhighlight %}


## Attention 
We have 3 learnable weight parameters - queries $\boldsymbol{Q}$, keys $\boldsymbol{K}$, values $\boldsymbol{V}$ each with dimensions $(d_q+1)x(d_q+1), (d_k+1)x(d_k+1),(d_v+1)x(d_v+1)$ respectively. Let $\boldsymbol{x}$ be a $(d_q+1)xn$ dimensional matrix Next apply a linear transformation of the embeddings $\boldsymbol{x}$ with all 3 weight matrices giving us $\boldsymbol{q},\boldsymbol{k},\boldsymbol{v}$. 
$$\begin{equation}
\boldsymbol{Q}\boldsymbol{x} = \boldsymbol{q}\\
\boldsymbol{K}\boldsymbol{x} = \boldsymbol{k}\\
\boldsymbol{V}\boldsymbol{x} = \boldsymbol{v}\\
\tag{2}\label{eq:qkv}
\end{equation}
$$

For simplicity sake, let us assume that every word embedding of a sentences is a key, keys = values and each word embedding is a query. This implies that $d_q=d_k$ .

$$Cosine\ Similarity\ \boldsymbol{e} = \frac{\boldsymbol{q}^T\boldsymbol{k}}{\sqrt{d_k}}\tag{3}\label{eq:cosine similarity}$$

Note that the cosine similarity has been scaled by $\frac{1}{\sqrt{d_k}}$. To gain an intuition as to why, compare the norm betwen a 2 element vector and a 3 element vector. e.g.
$$\begin{aligned}
&\sqrt{\begin{bmatrix}2 \\ 2\end{bmatrix} \bullet \begin{bmatrix}2 \\ 2\end{bmatrix}}=\sqrt{2^2+2^2}
&=2\sqrt{2}
\end{aligned}$$
$$\begin{aligned}
&\sqrt{\begin{bmatrix}2 \\ 2 \\ 2\end{bmatrix} \bullet \begin{bmatrix}2 \\2 \\ 2\end{bmatrix}}=\sqrt{2^2+2^2+2^2}
&=2\sqrt{3}
\end{aligned}$$
It is clear that as the dimensions $d$ grow,the norm scales by $\sqrt{d}$.

$$Attention\ Weights \ \boldsymbol{\alpha} = softmax(\frac{\boldsymbol{q}^T\boldsymbol{k}}{\sqrt{d_k}})\tag{4}\label{eq:attention weights}$$

$$Attention\ \boldsymbol{z}= softmax(\frac{\boldsymbol{q}^T\boldsymbol{k}}{\sqrt{d_k}})\boldsymbol{v} \tag{5}\label{eq:attention}$$

Let us define 2 helper functions <b><i>LinearTransform</i></b> and <b><i>softmax</i></b> as below. 
{% highlight julia %}
function LinearTransform(x,W,b)
    return W*x.+b
end

function softmax(x)
    x = x .- maximum(x)
    return exp.(x) ./ sum(exp.(x))
end
{% endhighlight %}


Then define a function called <b><i>Attention</i></b> as below.

{% highlight julia %}
# Attention block
function Attention(x,Q,Qb,K,Kb,V,Vb)
    # queries
    q = LinearTransform(x,Q,Qb)
    # keys
    k = LinearTransform(x,K,Kb) 
    # values
    v = LinearTransform(x,V,Vb)

    # Attention Weights
    α = AttentionWeights(x,q,k,v)

    # context vectors
    z = v * α' 
    return q,k,v,α,z
end
{% endhighlight %}

The first line creates a function called <b><i>Attention</i></b>, which takes in the word embeddings $\boldsymbol{x}$ and trainable parameters $\boldsymbol{Q,Qb,K,Kb,V,Vb}$ as arguments. Note that I have chosen to separate the Query $\boldsymbol{Q}$, Key $\boldsymbol{K}$, Value $\boldsymbol{V}$ weight matrices from their biases $\boldsymbol{Qb,Kb,Vb}$ for clarity. So $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V},\boldsymbol{x}$ are $(d_qxd_q),(d_qxd_q),(d_qxd_q),(d_qxn)$ dimensions.
{% highlight julia %}
    # queries
    q = LinearTransform(x,Q,Qb)
    # keys
    k = LinearTransform(x,K,Kb) 
    # values
    v = LinearTransform(x,V,Vb)

{% endhighlight %}
The next 3 lines applies a linear transformation to the word embeddings as per equation (\ref{eq:qkv}).
{% highlight julia %}
    # Attention Weights
    α = AttentionWeights(x,q,k,v)
{% endhighlight %}
This line calls on a function <b><i>AttentionWeights</i></b> which returns the attention weight matrix $\alpha$ as per equation (\ref{eq:attention weights}).
{% highlight julia %}
    # context vectors
    z = v * α' 
{% endhighlight %}
The last line implements equation (\ref{eq:attention}) and returns the context vectors. 
{% highlight julia %}
# Return Attention Weights
function AttentionWeights(x,q,k,v)
    # compute similarity between queries and keys (with scaling)
    e = q'*k/sqrt(length(q))

    # initialize attention weight matrix α with zeroes
    α = zeros(size(e))

    # normalize each similarity row with softmax
    for row in 1:size(e)[1]
        α[row,:] = softmax(e[row,:])
    end    
    return α
end
{% endhighlight %}
The <b><i>AttentionWeights</i></b> function firsts computes the unnormalized cosine similarity matrix $e$ between queries and keys in (\ref{eq:attention}), scaled by the square root of key dimensions. Then, we apply the softmax function along each row of the resulting matrix.

Now we have our attention building block! For more detailed explanation for the forward and backpropagation steps for attention, please refer to this [post](https://kcin96.github.io/notes/ml/2024/01/18/backpropagation.html).

## Minimal Architecture
To see attention in action, let us consider a minimal functional architecture with attention at the heart of it.

<svg width="1000" height="250">>

<rect x="110" y="40" width="20" height="85" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="130" y="40" width="20" height="85" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="150" y="40" width="20" height="85" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="110" y="160" width="20" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="130" y="160" width="20" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="150" y="160" width="20" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="110" y="80">x1</text>
<text x="130" y="80">x2</text>
<text x="150" y="80">x3</text>
<text x="135" y="145">+</text>
<text x="115" y="180">0</text>
<text x="135" y="180">1</text>
<text x="155" y="180">2</text>
<text x="90" y="25">Embeddings</text>
<text x="80" y="205">Postional embeddings</text>


<rect x="245" y="40" width="100" height="125" style="fill:pink;stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="260" y="100">Attention</text>

<rect x="395" y="40" width="100" height="125" style="fill:aqua;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<circle cx="420" cy="50" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="420" cy="70" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="420" cy="90" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="470" cy="50" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="470" cy="70" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<circle cx="470" cy="90" r="5" stroke="black" stroke-width="1" fill="white"></circle>
<polyline points="425,50 465,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,50 465,70" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,50 465,90" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,70 465,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,70 465,70" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,70 465,90" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,90 465,50" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,90 465,70" style="fill:none;stroke:black;stroke-width:1" />
<polyline points="425,90 465,90" style="fill:none;stroke:black;stroke-width:1" />
<text x="400" y="130">Feedforward</text>
<text x="415" y="150">network</text>

<rect x="540" y="40" width="60" height="125" style="fill:yellow;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="542" y="100">Pooling</text>

<rect x="640" y="40" width="80" height="125" style="fill:rgb(50,200,200);stroke-width:3;stroke:rgb(0,0,0)"></rect>
<text x="650" y="100">Softmax</text>

<rect x="800" y="60" width="25" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="800" y="85" width="25" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>
<rect x="800" y="110" width="25" height="25" style="fill:None;stroke-width:3;stroke:rgb(0,0,0)"></rect>

<text x="780" y="190">Prediction</text>
<text x="770" y="210">Probabilities</text>

<polyline points="170,95 240,95 220,85 240,95 220,105" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="345,95 395,95 380,85 395,95 380,105" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="495,95 540,95 530,85 540,95 530,105" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="600,95 640,95 630,85 640,95 630,105" style="fill:none;stroke:black;stroke-width:2" />
<polyline points="720,95 800,95 780,85 800,95 780,105" style="fill:none;stroke:black;stroke-width:2" />
</svg>

The architecture consists of 4 intermediate layers - Attention layer, Feed forward layer, Pooling layer, Softmax layer. For the input, we pass in a concatenation of word embeddings and the positional embeddings. At the final output layer, we obtain the classification probabilities. For this example, let us assume that we are doing sentiment analysis for a sentence. The sentiment labels are one-hot vectors - positive $[0,0,1]$, negative $[1,0,0]$, neutral $[0,1,0]$. So when we pass in an input sentence through the above architecture, we get the probabilities of whether our input sentence is either positive, negative or neutral.    

## Forward pass
Let us now create a function <b><i>Forwardprop</i></b> for the foward pass through the above architecture. The function takes in a word embedding matrix $\boldsymbol{x}$, the Attention parameters $\boldsymbol{Q,Q_b,K,K_b,V,V_b}$ and the Feedforward parameters $\boldsymbol{W,b}$ as arguments.
{% highlight julia %}
function forwardprop(x,Q,Qb,K,Kb,V,Vb,W,b)
    # Reshape from 1d to 2d
    if ndims(x)==1
        x = reshape(x,(:,1))  
    end
    x = vcat(x,Vector(range(0,size(x)[2]-1))') 
{% endhighlight %}
The <b><i>if</i></b> condition handles a sentence with just a single word by reshaping the 1D word embeddings to 2D. The <b><i>vcat</i></b> concatenates the word embeddings $x$ with its position embeddings (index position in the sentence).

{% highlight julia %}
    # Return attention values
    q,k,v,α,z = Attention(x,Q,Qb,K,Kb,V,Vb)
{% endhighlight %}
We next pass in the arguments to the <b><i>Attention</i></b> function as defined in [Attention](#attention), which returns the queries, keys, values, attention weights and context vectors.
{% highlight julia %}
    # Feed Forward layer
    f = FeedForward(z,W,b)  #shape: [3xn]
{% endhighlight %}
In the third layer, we pass the $z$ context vectors, currently a $d_vxn$ matrix through a feedforward layer. The <b><i>Feedforward</i></b> function is shown below.
{% highlight julia %}
function FeedForward(x,W,b)
    return LinearTransform(x,W,b) 
end
{% endhighlight %}
The feedforward layer transforms the context vectors $z$ to a $3xn$ matrix.
{% highlight julia %}
    # Average pooling. 
    p = sum(f,dims=2)/size(f)[2]  #shape: [3x1]
    
    # Softmax layer to get probabilities 
    p = softmax(p)

end
{% endhighlight %}
Now, before passing it through to the softmax function, we need to reduce the 3xn matrix to a 3x1 matrix. A simple naive way would be to just average the $n$ columns. Hence, we next pass $f$ through a   pooling layer to obtain a 3x1 vector before passing it through a softmax layer. The final values returns the probabilities of the sentiments. 
 
Putting it all together,
{% highlight julia %}
function FeedForward(x,W,b)
    return LinearTransform(x,W,b) 
end

#Forward propagate
function forwardprop(x,Q,Qb,K,Kb,V,Vb,W,b)
    # Reshape from 1d to 2d
    if ndims(x)==1
        x = reshape(x,(:,1))  
    end
    x = vcat(x,Vector(range(0,size(x)[2]-1))') 
    
    # Return attention values
    q,k,v,α,z = Attention(x,Q,Qb,K,Kb,V,Vb)

    # Feed Forward layer
    f = FeedForward(z,W,b)  #shape: [3xn]

    # Average pooling. 
    p = sum(f,dims=2)/size(f)[2]  #shape: [3x1]

    # Softmax layer to get probabilities 
    p = softmax(p)

    return x,q,k,v,α,z,p
end
{% endhighlight %}

## Backwards pass
This section dives into the details of computing gradients required for backpropagation.
#### Cross Entropy 
As we have a softmax function at the last layer, a good loss function to use is the cross entropy loss, defined below. 
Note that mean squared error can work poorly with softmax, explained [here](https://kcin96.github.io/notes/ml/2023/12/29/why-mean-squared-loss-works-poorly-with-softmax.html). 

{% highlight julia %}
function CrossEntropyLoss(z,x)
    return -sum(x.*log.(z))
end
{% endhighlight %}
Working our way backwards, we first compute the derivative of loss with respect to the softmax, which is just the prediction p - the training label y. 

$$\begin{aligned}
\frac{\partial{L}}{\partial{\boldsymbol{p}}}=\boldsymbol{p}-\boldsymbol{y}\\
\end{aligned}$$

{% highlight julia %}
function backprop(x,y,
                Q,Qb,K,Kb,V,Vb,W,b,
                q,k,v,α,z,p,
                η=.001)
    # Softmax gradient ∂L/∂σ
    ∂L_∂p = p-y    #shape: [3x1] 
{% endhighlight %}


#### Pooling
Recall that for the pooling layer, we averaged across all columns as below.

$$\begin{aligned}
\boldsymbol{f}&=\begin{bmatrix}\boldsymbol{f_1} \ \boldsymbol{f_2} \ \boldsymbol{f_3} \ \ldots \boldsymbol{f_n}
\end{bmatrix}\\
\boldsymbol{p}&=\frac{1}{n}\sum_{i=1}^{n}\boldsymbol{p_i}\\
&=\frac{1}{n}(\boldsymbol{f_1} + \boldsymbol{f_2} + \boldsymbol{f_3} + \ldots + \boldsymbol{f_n}) \\ 
\end{aligned}\tag{6}\label{eq:pooling}$$

The derivative of (\ref{eq:pooling}) gives us the gradients

$$\begin{aligned}
\frac{\partial{p}}{\partial{\boldsymbol{f}}}
&=\frac{1}{n}\begin{bmatrix}1 \ 1 \ \ldots 1\\ \end{bmatrix}
\end{aligned}$$

$$\begin{equation}
\frac{\partial{L}}{\partial{\boldsymbol{f}}}=\frac{\partial{L}}{\partial{\boldsymbol{p}}}\frac{\partial{\boldsymbol{p}}}{\partial{\boldsymbol{f}}}\\\tag{7}\label{eq:dLdf}
\end{equation} $$
{% highlight julia %}
    # Average pooling gradient ∂L/∂f
    ∂p_∂f = (1 ./size(z)[2] .*ones(1,size(z)[2]))
    ∂L_∂f = ∂L_∂p*∂p_∂f  #shape: [3xn] 
    
{% endhighlight %}
#### Feedforward
The local gradients of $\boldsymbol{f}$ are 
$$\begin{equation}\frac{\partial{\boldsymbol{f}}}{\partial{\boldsymbol{z}}}=\boldsymbol{W}\\ \tag{8}\label{eq:dfdz}\end{equation}$$

$$\begin{equation}
\frac{\partial{\boldsymbol{f}}}{\partial{\boldsymbol{W}}}=\boldsymbol{z}^T\\ \tag{9}\label{eq:dfdW}
\end{equation}$$

Using (\ref{eq:dLdf}) & (\ref{eq:dfdW})
$$\begin{equation}
\frac{\partial{L}}{\partial{\boldsymbol{W}}}=\frac{\partial{L}}{\partial{\boldsymbol{f}}}\frac{\partial{\boldsymbol{f}}}{\partial{\boldsymbol{W}}}\\ \tag{10}\label{eq:dLdW}
\end{equation}$$
$$\begin{equation}
\frac{\partial{L}}{\partial{\boldsymbol{b}}}=sumcol(\frac{\partial{L}}{\partial{\boldsymbol{f}}})\\ \tag{11}\label{eq:dLdb}
\end{equation}$$
We will need (\ref{eq:dLdW}) and (\ref{eq:dLdb}) later during the parameter update step.
{% highlight julia %}    
    # NN local gradients ∂f/∂z, ∂f/∂W
    ∂f_∂z = W  #shape: [3xd] 
    ∂f_∂W = z' #shape: [4xd]

    # NN gradients ∂L/∂W and ∂L/∂b
    ∂L_∂W = ∂L_∂f*∂f_∂W  #shape: [3xd]  
    ∂L_∂b = sum(∂L_∂f,dims=2)  #shape: [3x1] 

{% endhighlight %}

#### Context vector
Using (\ref{eq:dLdf}) & (\ref{eq:dfdz})
$$\begin{equation}
\frac{\partial{L}}{\partial{\boldsymbol{z}}}=\frac{\partial{L}}{\partial{\boldsymbol{f}}}\frac{\partial{\boldsymbol{f}}}{\partial{\boldsymbol{z}}}\\ \tag{12}\label{eq:dLdz}
\end{equation}$$
{% highlight julia %}    
    # Context vector gradients
    ∂L_∂z = (∂L_∂f'*∂f_∂z)' #shape: [dxn]  


{% endhighlight %}
The transpose operations ensures that the shape of the gradient matrix matches that of $\boldsymbol{z}$.

#### Attention
The below code computes the gradients for the attention layer. For more detailed explanations, please refer to this [post](https://kcin96.github.io/notes/ml/2024/01/18/backpropagation.html).
{% highlight julia %}
   # Attention gradients
    # Local value gradients ∂z/∂v, ∂v/∂V  
    ∂z_∂v = α  #shape: [nxn] 
    ∂v_∂V = x' #shape: [nxd]

    # Local attention weight gradients ∂z/∂α 
    ∂z_∂α = v  #shape: [dxn] 

    # Initialize ∂α/∂e to zeroes
    ∂α_∂e = zeros(size(α)[1],size(α)[2])  #shape: [nxn]

    # Derivative of softmax
    for k in 1:size(α)[1]
        for j in 1:size(α)[2]
            if j == k
                ∂α_∂e[j,k] = α[j]*(1-α[j]) 
            else
                ∂α_∂e[j,k] = -α[k]*α[j]
            end
        end
    end
    
    # Local query, key gradients ∂e_∂q, ∂e_∂k 
    ∂e_∂q, ∂e_∂k = k', q'  #shape: [nxd],[nxd] 
    ∂q_∂Q, ∂k_∂K = x', x'  #shape: [nxd],[nxd]  

    # Softmax gradients
    ∂L_∂α = ∂L_∂z'*∂z_∂α   #shape: [nxn]

    # Similarity score gradients
    ∂L_∂e = ∂L_∂α*∂α_∂e    #shape: [nxn] 

    # query gradients
    ∂L_∂q = ∂L_∂e*∂e_∂q  #shape: [nxd]
    # key gradients
    ∂L_∂k = ∂L_∂e'*∂e_∂k #shape: [nxd] 
    # values gradients
    ∂L_∂v = ∂L_∂z*∂z_∂v  #shape: [dxn]

    # Q,K,V parameter gradients 
    ∂L_∂Q = ∂L_∂q'*∂q_∂Q  #shape: [dxd]
    ∂L_∂K = ∂L_∂k'*∂k_∂K  #shape: [dxd]
    ∂L_∂V = ∂L_∂v*∂v_∂V   #shape: [dxd]

    ∂L_∂Qb = sum(∂L_∂q',dims=2)  #shape: [dx1]
    ∂L_∂Kb = sum(∂L_∂k',dims=2) #shape: [dx1]
    ∂L_∂Vb = sum(∂L_∂v,dims=2)  #shape: [dx1]

{% endhighlight %}

#### Update step
The first block of code below initializes new parameters to the current ones. Then, using the gradients computed previously for attention and equations (\ref{eq:dLdW}) and (\ref{eq:dLdb}), the new $Q_{new},Qb_{new},K_{new},Kb_{new},V_{new},Vb_{new},W_{new},b_{new}$ can be updated using SGD. 

{% highlight julia %}
    # Update Attention parameters
    # Initialize new parameter matrices with current parameters
    Q_new = Q
    Qb_new = Qb
    K_new = K 
    Kb_new = Kb
    V_new = V
    Vb_new = Vb
    W_new = W
    b_new = b

    # Update all trainable parameters with SGD
    Q_new = Q_new .- η * ∂L_∂Q
    Qb_new = Qb_new .- η * ∂L_∂Qb
    
    K_new = K_new .- η * ∂L_∂K   
    Kb_new = Kb_new .- η * ∂L_∂Kb
    
    V_new = V_new .- η * ∂L_∂V 
    Vb_new = Vb_new .- η * ∂L_∂Vb

    W_new = W_new .- η * ∂L_∂W
    b_new = b_new .- η * ∂L_∂b
{% endhighlight %}


#### backprop code
The full <b><i>backprop</i></b> function is as below.
{% highlight julia %}
# Backpropagate
function backprop(x,y,
                Q,Qb,K,Kb,V,Vb,W,b,
                q,k,v,α,z,p,
                η=.001)
    # Softmax gradient ∂L/∂σ
    ∂L_∂p = p-y    #shape: [3x1] 

    # Average pooling gradient ∂L/∂f
    ∂p_∂f = (1 ./size(z)[2] .*ones(1,size(z)[2]))
    ∂L_∂f = ∂L_∂p*∂p_∂f  #shape: [3xn] 
    
    # NN local gradients ∂f/∂z, ∂f/∂W
    ∂f_∂z = W  #shape: [3xd] 
    ∂f_∂W = z' #shape: [4xd]

    # NN gradients ∂L/∂W and ∂L/∂b
    ∂L_∂W = ∂L_∂f*∂f_∂W  #shape: [3xd]  
    ∂L_∂b = sum(∂L_∂f,dims=2)  #shape: [3x1] 

    # Context vector gradients
    ∂L_∂z = (∂L_∂f'*∂f_∂z)' #shape: [dxn]  

    # Attention gradients
    # Local value gradients ∂z/∂v, ∂v/∂V  
    ∂z_∂v = α  #shape: [nxn] 
    ∂v_∂V = x' #shape: [nxd]

    # Local attention weight gradients ∂z/∂α 
    ∂z_∂α = v  #shape: [dxn] 

    # Initialize ∂α/∂e to zeroes
    ∂α_∂e = zeros(size(α)[1],size(α)[2])  #shape: [nxn]

    # Derivative of softmax
    for k in 1:size(α)[1]
        for j in 1:size(α)[2]
            if j == k
                ∂α_∂e[j,k] = α[j]*(1-α[j]) 
            else
                ∂α_∂e[j,k] = -α[k]*α[j]
            end
        end
    end
    
    # Local query, key gradients ∂e_∂q, ∂e_∂k 
    ∂e_∂q, ∂e_∂k = k', q'  #shape: [nxd],[nxd] 
    ∂q_∂Q, ∂k_∂K = x', x'  #shape: [nxd],[nxd]  

    # Softmax gradients
    ∂L_∂α = ∂L_∂z'*∂z_∂α   #shape: [nxn]

    # Similarity score gradients
    ∂L_∂e = ∂L_∂α*∂α_∂e    #shape: [nxn] 

    # query gradients
    ∂L_∂q = ∂L_∂e*∂e_∂q  #shape: [nxd]
    # key gradients
    ∂L_∂k = ∂L_∂e'*∂e_∂k #shape: [nxd] 
    # values gradients
    ∂L_∂v = ∂L_∂z*∂z_∂v  #shape: [dxn]

    # Q,K,V parameter gradients 
    ∂L_∂Q = ∂L_∂q'*∂q_∂Q  #shape: [dxd]
    ∂L_∂K = ∂L_∂k'*∂k_∂K  #shape: [dxd]
    ∂L_∂V = ∂L_∂v*∂v_∂V   #shape: [dxd]

    ∂L_∂Qb = sum(∂L_∂q',dims=2)  #shape: [dx1]
    ∂L_∂Kb = sum(∂L_∂k',dims=2) #shape: [dx1]
    ∂L_∂Vb = sum(∂L_∂v,dims=2)  #shape: [dx1]

    # Update Attention parameters
    # Initialize new parameter matrices with current parameters
    Q_new = Q
    Qb_new = Qb
    K_new = K 
    Kb_new = Kb
    V_new = V
    Vb_new = Vb
    W_new = W
    b_new = b

    # Update all trainable parameters with SGD
    Q_new = Q_new .- η * ∂L_∂Q
    Qb_new = Qb_new .- η * ∂L_∂Qb
    
    K_new = K_new .- η * ∂L_∂K   
    Kb_new = Kb_new .- η * ∂L_∂Kb
    
    V_new = V_new .- η * ∂L_∂V 
    Vb_new = Vb_new .- η * ∂L_∂Vb

    W_new = W_new .- η * ∂L_∂W
    b_new = b_new .- η * ∂L_∂b

    return Q_new,Qb_new,K_new,Kb_new,V_new,Vb_new,W_new,b_new
end
{% endhighlight %}

## Sentiment data
For our crude sentiment example, let us create a small sentiment dataset with first column containing the sentiment and second column containing the text. The contents for this "small.csv" are

{% highlight csv %}
sentiments,cleaned_review
positive,i love this speaker
negative,this mouse was waste of money it broke after using it 
positive,great product 
neutral,it meets my need 
negative,the volume on this is lower than my flip big disappointment
positive,the best
negative,little disappointed and it only comes with charging cord not wall plug 
positive,super
neutral,caught this on sale and for the price the quality is unbeatable 
neutral,relax on my home
negative,the battery died in week doesn charge 
negative,product is sub par so many things bad is that enough for me to submit it terrible 
negative,had an older jbl portable speaker the charge is way more expensive and larger than my old one but not any louder was disappointed in that
neutral,easy to carry 
neutral,the sound that comes out of this thing is incredible 
positive,its strong clear and great
positive,loved it awesome
positive,i am happy with this product
negative,i am not happy with this
negative,it stopped working my is very disappointed
positive,i love the sound quality this is great product for the price 
negative,bad audio input
negative,pretty decent but not the best for box
negative,does not work for ps bad quality not recommend
neutral,the clear sound
negative,my son only used this gaming headset for few months and the mic already quit working very disappointed 
negative,the usb plug in is not long enough to connect to the playstation the cord is so long but splits into two six inch cords one goes into the playstation the other goes into the controller what the hell are you supposed to do
negative,they are uncomfortable and seem really fragile 
positive,son loved them 
neutral,thanks
negative,very frustrating as they both broke
negative,the color and shape are very nice but the mouse and the packaging arrived very dirty
negative,product lags a lot
positive,this is great service
positive,good quality product
negative,very frustrating bad quality
positive,we loved the soft texture
neutral,no sound
positive,good price
{% endhighlight %}

We next import in the packages CSV and DataFrames to allow us to load in the csv file.
{% highlight julia %}
using CSV, DataFrames
tb = CSV.read("small.csv",DataFrame)
{% endhighlight %}

To handle the case where the word in a sentence is not in our embeddings dictionary, for simplicity, we will just drop it using the following <b><i>remove_nid</i></b> function.
{% highlight julia %}
# Removes words that are not in dictionary
function remove_nid(sentence)
    sen = []
    if !ismissing(sentence)
        for i in word_tokeniser(sentence)
            try get_embeddings(i)
                push!(sen,i)
            catch e
            end
        end
    end
    return sen
end

{% endhighlight %}


## Training
For training, we have the below <b><i>train</i></b> function which
* calls on <b><i>forwardprop</i></b> for the forward propagate step
* computes the Cross Entropy Loss with <b><i>CrossEntropyLoss(p,y)</i></b>
* backpropagates and returns the updated parameters with <b><i>backprop(x,y,train_params...,q,k,v,α,z,p)</i></b>

Note that we absorbed the train parameters with the julia splat (...) notation to keep the function readable.
{% highlight julia %}
# Train step
function train(x,y,train_params...)
    x,q,k,v,α,z,p = forwardprop(x,train_params...) 
    CEloss = CrossEntropyLoss(p,y)
    train_params = backprop(x,y,train_params...,q,k,v,α,z,p)
    return train_params...,CEloss
end
{% endhighlight %}

The below code
* initializes small and random training parameters 
* creates a dictionary that maps sentiments (positive, negative, neutral) to one-hot vectors
* processes our input sentences
* trains our minimal architecture
* returns loss

{% highlight julia %}
# main 
using Random
# Random seed for reproducibility
rng = MersenneTwister(12);

# Initialize small random parameter values
Q = randn(rng, (51, 51))/100
Qb = zeros(51,1)
K = randn(rng, (51, 51))/100
Kb = zeros(51,1)
V = K
Vb = zeros(51,1)
W = randn(rng, (3, 51))/100
b = zeros(3,1) 

# Sentiment dictionary that converts sentiment
# text into one-hot labels
sent_dict = Dict("positive"=>[0,0,1],"negative"=>[1,0,0],"neutral"=>[0,1,0])

#training
for epoch=1:1000
    total_l = 0   #total loss
    for idx in 1:nrow(tb)
        x_em = []
        l = 0   #current loss
        sen = tb[idx,"cleaned_review"]  #gets sentence
        sen = remove_nid(sen)  #remove words not in dictionary
        if length(sen)!=0
            for i in (sen)
                if length(x_em) == 0
                    x_em = get_embeddings(i)
                else 
                    #Concatenate word embeddings along columns
                    x_em = hcat(x_em,get_embeddings(i)) 
                end
            end
            #One hot vector sentiment
            y = sent_dict[tb[idx,"sentiments"]]
            #Update parameters
            Q,Qb,K,Kb,V,Vb,W,b,l = train(x_em,y,Q,Qb,K,Kb,V,Vb,W,b)
        end
        total_l += l
    end
    println("Total loss:", total_l/nrow(tb))
end
{% endhighlight %}

To see the queries, keys, values being trained, we will deliberately not update the feedforward network parameters by commenting out the update steps for $\boldsymbol{W,b}$ in the <b><i>backprop</i></b> function.
{% highlight julia %}
    W_new = W_new #.- η * ∂L_∂W
    b_new = b_new #.- η * ∂L_∂b

{% endhighlight %}

## Results
We are interested to see the resulting attention weights after training. To help us visualize that, we first import the julia Plots package.

{% highlight julia %}
using Plots
{% endhighlight %}

We then write a function <b><i>evaluate_model</i></b>  which takes in a sentence <b>sen</b>, removes words not in our dictionary, transforms each word into embeddings and then forward propagating them to obtain the attention weights. With the attention weights, we can now plot a heatmap with the Plots package.

{% highlight julia %}
# Evaluates the sentiment given a sentence as input
function evaluate_model(sen)
    x_em = []
    sen = remove_nid(sen)
    for i in (sen)
        if length(x_em) == 0
            x_em = get_embeddings(i)
        else 
            x_em = hcat(x_em,get_embeddings(i))
        end
    end

    α = forwardprop(x_em,Q,Qb,K,Kb,V,Vb,W,b)[5]

    # plot heatmap of α
    heatmap(sen,sen,α,clims=(0,1),aspect_ratio=1,color=:deepsea,
            title="Attention weights α",grid="off")
    
end
{% endhighlight %}
When we run
{% highlight julia %}
evaluate_model("very sad as they both fail")
{% endhighlight %} 
We can see from the below heatmap that the attention weights are "focusing" in on certain words.

![Image 1]({{site.baseurl}}/assets/images/attention-from-scratch/attentionweights1.svg)

{% highlight julia %}
evaluate_model("he loved that plug with good price ")
{% endhighlight %} 

![Image 2]({{site.baseurl}}/assets/images/attention-from-scratch/attentionweights2.svg)
{% highlight julia %}
evaluate_model("terrible quality for this price")
{% endhighlight %} 

![Image 3]({{site.baseurl}}/assets/images/attention-from-scratch/attentionweights3.svg)
{% highlight julia %}
evaluate_model("i love this fantastic product")
{% endhighlight %} 

![Image 4]({{site.baseurl}}/assets/images/attention-from-scratch/attentionweights4.svg)
{% highlight julia %}
evaluate_model("easy to move around")
{% endhighlight %} 

![Image 5]({{site.baseurl}}/assets/images/attention-from-scratch/attentionweights5.svg)

## Full Code
Putting it all together 
{% highlight julia %}
using CSV, DataFrames
using Plots
using Embeddings
using Random

tb = CSV.read("small.csv",DataFrame)

const embtable = load_embeddings(GloVe{:en},1,max_vocab_size=10000)
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

# Returns embeddings for word
function get_embeddings(word)
    return embtable.embeddings[:,get_word_index[word]]
end

# Splits sentence into a vector of words
function word_tokeniser(sentence)
    return split(sentence," ")
end

# Softmax function
function softmax(x)
    x = x .- maximum(x)
    return exp.(x) ./ sum(exp.(x))
end

# Cross Entropy Loss
function CrossEntropyLoss(z,x)
    return -sum(x.*log.(z))
end

# Linear Transformation
function LinearTransform(x,W,b)
    return W*x.+b
end

# Feedforward network
function FeedForward(x,W,b)
    return LinearTransform(x,W,b) 
end

# Return Attention Weights
function AttentionWeights(x,q,k,v)
    # compute similarity between queries and keys (with scaling)
    e = q'*k/sqrt(length(q))

    # initialize attention weight matrix α with zeroes
    α = zeros(size(e))

    # normalize each similarity row with softmax
    for row in 1:size(e)[1]
        α[row,:] = softmax(e[row,:])
    end    
    return α
end

# Attention block
function Attention(x,Q,Qb,K,Kb,V,Vb)
    # queries
    q = LinearTransform(x,Q,Qb)
    # keys
    k = LinearTransform(x,K,Kb) 
    # values
    v = LinearTransform(x,V,Vb)

    # Attention Weights
    α = AttentionWeights(x,q,k,v)

    # context vectors
    z = v * α' 
    return q,k,v,α,z
end

# Forward propagate
function forwardprop(x,Q,Qb,K,Kb,V,Vb,W,b)
    # Reshape from 1d to 2d
    if ndims(x)==1
        x = reshape(x,(:,1))  
    end
    x = vcat(x,Vector(range(0,size(x)[2]-1))') 
    
    # Return attention values
    q,k,v,α,z = Attention(x,Q,Qb,K,Kb,V,Vb)

    # Feed Forward layer
    f = FeedForward(z,W,b)  #shape: [3xn]

    # Average pooling. 
    p = sum(f,dims=2)/size(f)[2]  #shape: [3x1]

    # Softmax layer to get probabilities 
    p = softmax(p)

    return x,q,k,v,α,z,p
end

# Train step
function train(x,y,train_params...)
    x,q,k,v,α,z,p = forwardprop(x,train_params...) 
    CEloss = CrossEntropyLoss(p,y)
    train_params = backprop(x,y,train_params...,q,k,v,α,z,p)
    return train_params...,CEloss
end

# Backpropagate
function backprop(x,y,
                Q,Qb,K,Kb,V,Vb,W,b,
                q,k,v,α,z,p,
                η=.001)
    # Softmax gradient ∂L/∂σ
    ∂L_∂p = p-y    #shape: [3x1] 

    # Average pooling gradient ∂L/∂f
    ∂p_∂f = (1 ./size(z)[2] .*ones(1,size(z)[2]))
    ∂L_∂f = ∂L_∂p*∂p_∂f  #shape: [3xn] 
    
    # NN local gradients ∂f/∂z, ∂f/∂W
    ∂f_∂z = W  #shape: [3xd] 
    ∂f_∂W = z' #shape: [4xd]

    # NN gradients ∂L/∂W and ∂L/∂b
    ∂L_∂W = ∂L_∂f*∂f_∂W  #shape: [3xd]  
    ∂L_∂b = sum(∂L_∂f,dims=2)  #shape: [3x1] 

    # Context vector gradients
    ∂L_∂z = (∂L_∂f'*∂f_∂z)' #shape: [dxn]  

    # Attention gradients
    # Local value gradients ∂z/∂v, ∂v/∂V  
    ∂z_∂v = α  #shape: [nxn] 
    ∂v_∂V = x' #shape: [nxd]

    # Local attention weight gradients ∂z/∂α 
    ∂z_∂α = v  #shape: [dxn] 

    # Initialize ∂α/∂e to zeroes
    ∂α_∂e = zeros(size(α)[1],size(α)[2])  #shape: [nxn]

    # Derivative of softmax
    for k in 1:size(α)[1]
        for j in 1:size(α)[2]
            if j == k
                ∂α_∂e[j,k] = α[j]*(1-α[j]) 
            else
                ∂α_∂e[j,k] = -α[k]*α[j]
            end
        end
    end
    
    # Local query, key gradients ∂e_∂q, ∂e_∂k 
    ∂e_∂q, ∂e_∂k = k', q'  #shape: [nxd],[nxd] 
    ∂q_∂Q, ∂k_∂K = x', x'  #shape: [nxd],[nxd]  

    # Softmax gradients
    ∂L_∂α = ∂L_∂z'*∂z_∂α   #shape: [nxn]

    # Similarity score gradients
    ∂L_∂e = ∂L_∂α*∂α_∂e    #shape: [nxn] 

    # query gradients
    ∂L_∂q = ∂L_∂e*∂e_∂q  #shape: [nxd]
    # key gradients
    ∂L_∂k = ∂L_∂e'*∂e_∂k #shape: [nxd] 
    # values gradients
    ∂L_∂v = ∂L_∂z*∂z_∂v  #shape: [dxn]

    # Q,K,V parameter gradients 
    ∂L_∂Q = ∂L_∂q'*∂q_∂Q  #shape: [dxd]
    ∂L_∂K = ∂L_∂k'*∂k_∂K  #shape: [dxd]
    ∂L_∂V = ∂L_∂v*∂v_∂V   #shape: [dxd]

    ∂L_∂Qb = sum(∂L_∂q',dims=2)  #shape: [dx1]
    ∂L_∂Kb = sum(∂L_∂k',dims=2) #shape: [dx1]
    ∂L_∂Vb = sum(∂L_∂v,dims=2)  #shape: [dx1]

    # Update Attention parameters
    # Initialize new parameter matrices with current parameters
    Q_new = Q
    Qb_new = Qb
    K_new = K 
    Kb_new = Kb
    V_new = V
    Vb_new = Vb
    W_new = W
    b_new = b

    # Update all trainable parameters with SGD
    Q_new = Q_new .- η * ∂L_∂Q
    Qb_new = Qb_new .- η * ∂L_∂Qb
    
    K_new = K_new .- η * ∂L_∂K   
    Kb_new = Kb_new .- η * ∂L_∂Kb
    
    V_new = V_new .- η * ∂L_∂V 
    Vb_new = Vb_new .- η * ∂L_∂Vb

    W_new = W_new #.- η * ∂L_∂W
    b_new = b_new #.- η * ∂L_∂b

    return Q_new,Qb_new,K_new,Kb_new,V_new,Vb_new,W_new,b_new
end

# Removes words that are not in dictionary
function remove_nid(sentence)
    sen = []
    if !ismissing(sentence)
        for i in word_tokeniser(sentence)
            try get_embeddings(i)
                push!(sen,i)
            catch e
            end
        end
    end
    return sen
end

# Evaluates the sentiment given a sentence as input
function evaluate_model(sen)
    x_em = []
    sen = remove_nid(sen)
    for i in (sen)
        if length(x_em) == 0
            x_em = get_embeddings(i)
        else 
            x_em = hcat(x_em,get_embeddings(i))
        end
    end

    α = forwardprop(x_em,Q,Qb,K,Kb,V,Vb,W,b)[5]

    # plot heatmap of α
    heatmap(sen,sen,α,clims=(0,1),aspect_ratio=1,color=:deepsea,
            title="Attention weights α",grid="off")
    
end

# main 

# Random seed for reproducibility
rng = MersenneTwister(12);

# Initialize small random parameter values
Q = randn(rng, (51, 51))/100
Qb = zeros(51,1)
K = randn(rng, (51, 51))/100
Kb = zeros(51,1)
V = K
Vb = zeros(51,1)
W = randn(rng, (3, 51))/100
b = zeros(3,1) 

# Sentiment dictionary that converts sentiment
# text into one-hot labels
sent_dict = Dict("positive"=>[0,0,1],"negative"=>[1,0,0],"neutral"=>[0,1,0])

#training
for epoch=1:1000
    total_l = 0   #total loss
    for idx in 1:nrow(tb)
        x_em = []
        l = 0   #current loss
        sen = tb[idx,"cleaned_review"]  #gets sentence
        sen = remove_nid(sen)  #remove words not in dictionary
        if length(sen)!=0
            for i in (sen)
                if length(x_em) == 0
                    x_em = get_embeddings(i)
                else 
                    #Concatenate word embeddings along columns
                    x_em = hcat(x_em,get_embeddings(i)) 
                end
            end
            #One hot vector sentiment
            y = sent_dict[tb[idx,"sentiments"]]
            #Update parameters
            Q,Qb,K,Kb,V,Vb,W,b,l = train(x_em,y,Q,Qb,K,Kb,V,Vb,W,b)
        end
        total_l += l
    end
    println("Total loss:", total_l/nrow(tb))
end

# vizualize attention weights
evaluate_model("very sad as they both fail")

{% endhighlight %} 

And that is attention from scratch!

## References
1. [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html)
2. [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/)
3. [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf)
4. [https://kcin96.github.io/notes/ml/2024/01/18/backpropagation.html]((https://kcin96.github.io/notes/ml/2024/01/18/backpropagation.html))
5. [https://kcin96.github.io/notes/ml/2023/12/29/why-mean-squared-loss-works-poorly-with-softmax.html](https://kcin96.github.io/notes/ml/2023/12/29/why-mean-squared-loss-works-poorly-with-softmax.html)
6. [Attention Is All You Need](https://arxiv.org/abs/1706.03762v7)
7. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: [Global Vectors for Word Representation.](https://nlp.stanford.edu/pubs/glove.pdf)