---
layout: sidebar
title:  "Python vs Julia Cheatsheet"
date:   2023-11-04 21:16:00 +0100
categories: [Python, Julia]

---
* toc
{:toc}


## Comments
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
#This is a single line comment.
{% endhighlight %}
{% highlight python %}
"""
This is a multiline comment.
Enclose comments between triple 
single quotes ''' ''' or 
double quotes """ """. 
""" 
{% endhighlight %}
</td>
<td>
{% highlight julia %}
#This is a single line comment.
{% endhighlight %}
{% highlight julia %}
#=
This is a multiline comment.
Enclose comments between "#=" and "=#". 
=#
{% endhighlight %}
</td>
</tr>
</table>


## Casting
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
float(9)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
convert(Float64,9)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td><code>9.0</code></td>
<td><code>9.0</code></td>
</tr>
</table>


## Null 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
None
{% endhighlight %}
</td>
<td>
{% highlight julia %}
nothing
{% endhighlight %}
</td>
</tr>
</table>


## Tuples 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
tup = (4.5,"python",7e-6,None)
tup
tup[2]
tup[:3]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
tup = (4.5,"julia",7e-6,nothing)
tup
tup[2]
tup[:3]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
(4.5, 'python', 7e-06, None)
7e-06
(4.5, 'python', 7e-06)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
(4.5, "julia", 7.0e-6, nothing)
"julia"
7.0e-6
{% endhighlight %}
</td>
</tr>
</table>

## Functions
### Function 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
def f(x,y): 
    return x+y
def g(x,y):
    return x**3%y
f(-1,2)
g(7,3)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
f(x,y) = x+y
function g(x,y)
    return x^3%y
end
f(-1,2)
g(7,3)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
1
1
{% endhighlight %}
</td>
<td>
{% highlight julia %}
1
1
{% endhighlight %}
</td>
</tr>
</table>


### Lambda functions/Anonymous functions
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
list(map(lambda x:x**2-3, [1,2,3]))
{% endhighlight %}
</td>
<td>
{% highlight julia %}
map(x -> x^2-3, [1,2,3])
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
[-2, 1, 6]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{Int64}: 
-2
1 
6
{% endhighlight %}
</td>
</tr>
</table>


### Function arguments - Default
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
def f(x,*,y=6,z=-2):
    return x+y+z 
def g(x,y=6,z=-2):
    return x+y+z
f(5)
f(5,y=7,z=3)
g(5,y=7,z=3)
g(2)
g(2,4,6)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
f(x;y=6,z=-2)=x+y+z
g(x,y=2,z=2)=x+y+z
f(5) 
f(5,y=7,z=3)
f(5;y=7,z=3)
g(2)
g(2,4,6)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
9
15
15
6
12
{% endhighlight %}
</td>
<td>
{% highlight julia %}
9
15
15
6
12
{% endhighlight %}
</td>
</tr>
</table>


## Arrays
### Array/Vector
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array([1,2,3])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1,2,3]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
np.array([1,2,3])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{Int64}:
1
2
3
{% endhighlight %}
</td>
</tr>
</table>


### Array - append
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.append(np.array([23,5]),3)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
push!([23,5],3)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([23,  5,  3])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{Int64}:
23
5
3
{% endhighlight %}
</td>
</tr>
</table>


### Array - concatenation
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.concatenate([[2,4],[35,3543,3,54]])
{% endhighlight %}
</td>
<td>
1. vcat 
{% highlight julia %}
vcat([2,4],[35,3543,3,54])
{% endhighlight %}
2. reduce and vcat
{% highlight julia %}
reduce(vcat,([[2,4],[35,3543,3,54]]))
{% endhighlight %}
3. append!
{% highlight julia %}
append!([2,4],[35,3543,3,54])
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([2,4,35,3543,3,54])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
6-element Vector{Int64}:
2
4
35
3543
3
34
{% endhighlight %}
</td>
</tr>
</table>

### Array - Slicing
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
[1, 5, 6, 7][1:4]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 5 6 7][2:4]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
[5, 6, 7]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{Int64}:
 5
 6
 7
{% endhighlight %}
</td>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
#slice index [start:stop:step]
np.array(range(1,6))[0::2]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
#slice index [start:step:stop]
Array(1:5)[1:2:end]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([1, 3, 5])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{Int64}:
 1
 3
 5
{% endhighlight %}
</td>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array(range(1,6))[-1]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
Array(1:5)[end]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
5
{% endhighlight %}
</td>
<td>
{% highlight julia %}
 5
{% endhighlight %}
</td>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array(range(1,7))[1:-1:2]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
Array(1:6)[2:2:end-1]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([2, 4])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
 2-element Vector{Int64}:
 2
 4
{% endhighlight %}
</td>
</tr>
</table>

### Array - Reshape
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
x = np.array([1,2,3]) #1d array,
                      # shape: (3,)
x = x.reshape(-1,1) #3x1 2d array, 
                    #shape: (3,1)
x
{% endhighlight %}
</td>
<td>
{% highlight julia %}
x = [1,2,3]   #3 element Vector,
              #shape: (3,)
x = reshape(x,(:,1))  #3x1 Matrix,
                      #shape: (3,1)
x
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[1],
       [2],
       [3]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3×1 Matrix{Int64}:
 1
 2
 3
{% endhighlight %}
</td>
</tr>
</table>


## Matrices
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array([[1,2,3]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 2 3]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[1, 2, 3]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
1×3 Matrix{Int64}:
1  2  3
{% endhighlight %}
</td>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array([[1,2,3],[4,5,6]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 2 3;4 5 6]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[1, 2, 3],       
       [4, 5, 6]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
2×3 Matrix{Int64}:
1  2  3
4  5  6
{% endhighlight %}
</td>
</tr>
</table>


### Matrices - Transpose/Adjoint
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array([[1,2,3]]).T
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 2 3]'
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[1],
       [2],
       [3]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3×1 adjoint(::Matrix{Int64}) with eltype Int64:
1
2
3
{% endhighlight %}
</td>
</tr>
</table>


### Matrix muliplication
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
np.array([[1,2],[3,4]])@np.array([[4,5],[6,7]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 2;3 4]*[4 5;6 7]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[16, 19],
       [36, 43]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
2×2 Matrix{Int64}:
16  19
36  43
{% endhighlight %}
</td>
</tr>
</table>

### Broadcasting
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
import numpy as np
np.array([[1,2,3],[3,4,5]])-np.array([6,7,8])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
[1 2 3; 3 4 5] .- [6 7 8]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
array([[-5, -5, -5],
       [-3, -3, -3]])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
2×3 Matrix{Int64}:
 -5  -5  -5
 -3  -3  -3
{% endhighlight %}
</td>
</tr>
</table>

## Dictionaries 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
{'a':2,'b':3}
{% endhighlight %}
</td>
<td>
{% highlight julia %}
Dict('a'=>2,'b'=>3)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
{'a': 2, 'b': 3}
{% endhighlight %}
</td>
<td>
{% highlight julia %}
Dict{Char, Int64} with 2 entries:
'a' => 2
'b' => 3
{% endhighlight %}
</td>
</tr>
</table>


## Strings
### Strings or Chars?
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
'a' == "a"
type(''),type("") 
{% endhighlight %}

</td>
<td>
{% highlight julia %}
'a' == "a"
typeof('a'),typeof("")
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
True
(<class 'str'>, <class 'str'>)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
false
(Char, String)
{% endhighlight %}
</td>
</tr>
</table>


### Strings - Indexing
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
s = "python is 0 indexed" 
s[0],s[-1],s[1:9] 
{% endhighlight %}
</td>
<td>
{% highlight julia %}
s = "julia is 1 indexed" 
s[1],s[end],s[1:9]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
('p', 'd', 'ython is')
{% endhighlight %}
</td>
<td>
{% highlight julia %}
('j', 'd', "julia is ")
{% endhighlight %}
</td>
</tr>
</table>


### Strings - Concatenation
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
1. .join
{% highlight python %}
"".join(["concat"," 2"," strings"])
{% endhighlight %}
2. + operator
{% highlight python %}
"concat"+" 2"+" strings"
{% endhighlight %}
</td>
<td>
1. string
{% highlight julia %}
string("concat"," 2"," strings")
{% endhighlight %}
2. * operator
{% highlight julia %}
"concat"*" 2"*" strings"
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
'concat 2 strings'
{% endhighlight %}
</td>
<td>
{% highlight julia %}
"concat 2 strings"
{% endhighlight %}
</td>
</tr>
</table>


### Strings - Length
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
len("python")
{% endhighlight %}
</td>
<td>
{% highlight julia %}
length("julia")
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
6
{% endhighlight %}
</td>
<td>
{% highlight julia %}
5
{% endhighlight %}
</td>
</tr>
</table>


### Strings - Regex
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
import re
re.findall("[a-zA-Z]p+","Apps that 
capture the spacious extent of 
the opera.")
{% endhighlight %}
</td>
<td>
{% highlight julia %}
collect(eachmatch(r"[a-zA-Z]p+","Apps 
that capture the spacious extent of 
the opera."))
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
['App', 'ap', 'sp', 'op']
{% endhighlight %}
</td>
<td>
{% highlight julia %}
4-element Vector{RegexMatch}:
RegexMatch("App")
RegexMatch("ap")
RegexMatch("sp")
RegexMatch("op")
{% endhighlight %}
</td>
</tr>
</table>


## f strings/String interpolation
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
num = 9.0
print(f"This substitutes {num}")
{% endhighlight %}
</td>
<td>
{% highlight julia %}
num = 9.0
println("This substitutes $num")
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td><code>This substitutes 9.0</code></td>
<td><code>This substitutes 9.0</code></td>
</tr>
</table>


## Conditionals 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
if x>0:
    print("x greater than 0.")
elif x<0:
    print("x less than 0.")
else:
    print("x equal to 0.")
{% endhighlight %}
</td>
<td>
{% highlight julia %}
if x>0
   print("x greater than 0.")
elseif x<0
    print("x less than 0.")
else
   print("x equal to 0.")
end
{% endhighlight %}
</td>
</tr>
</table>


## Logic 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
5>0 and 6>0 and 6>-9
0==0 or 7>-9
not True
{% endhighlight %}
</td>
<td>
{% highlight julia %}
5>0 && 6>0 && 6>-9
0==0 || 7>-9 
!true
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
True
True
False
{% endhighlight %}
</td>
<td>
{% highlight julia %}
true
true
false
{% endhighlight %}
</td>
</tr>
</table>


## Loops
### While loops
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
x = -7
while True:
    print(x)
    if x == -5:
        break
    x += 1
{% endhighlight %}
</td>
<td>
{% highlight julia %}
x = -7
while x != -5
    println(x)
    global x += 1
end
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
-7
-6
-5
{% endhighlight %}
</td>
<td>
{% highlight julia %}
-7
-6
{% endhighlight %}
</td>
</tr>
</table>


### For loops
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
for i in range(-3,3):
    if i%2 == 0:
       print(i)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
for i = -3:3
    if i%2 == 0
       println(i)
    end
end
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
-2
0
2
{% endhighlight %}
</td>
<td>
{% highlight julia %}
-2
0
2
{% endhighlight %}
</td>
</tr>
</table>

## List comprehension 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
 X = ['a','b','c']
 Y = range(0,3)
[(x,y) for x in X for y in Y]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
X = ['a','b','c']
Y = 1:3
[(x,y) for x in X, y in Y]
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
[('a', 0), ('a', 1), ('a', 2), 
('b', 0), ('b', 1), ('b', 2), 
('c', 0), ('c', 1), ('c', 2)]
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3×3 Matrix{Tuple{Char, Int64}}:
 ('a', 1)  ('a', 2)  ('a', 3)
 ('b', 1)  ('b', 2)  ('b', 3)
 ('c', 1)  ('c', 2)  ('c', 3)
{% endhighlight %}
</td>
</tr>
</table>

## Exceptions
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
import math
def f(x):
    try: math.sqrt(x)
    except: print("x should be non-negative.")
f(-9)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
function f(x)
    try
        sqrt(x)
    catch e
        println("x should be non-negative.")
    end
end
f(-9)
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
x should be non-negative.
{% endhighlight %}
</td>
<td>
{% highlight julia %}
x should be non-negative.
{% endhighlight %}
</td>
</tr>
</table>


## Assertions

| |Python | Julia |
| --- | --- | --- |
|Code|<code>assert 8 == 1</code> | <code>@assert 8 == 1</code>|
|Output|<code>AssertionError</code> | <code>ERROR: AssertionError: 8 == 1</code>|


## Counting 
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
from collections import Counter
Counter([0.5, 0.5, 1, 4, 3, 3, 2, 1])
{% endhighlight %}
</td>
<td>
{% highlight julia %}
using StatsBase
countmap([0.5 0.5 1 4 3 3 2 1])
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
Counter({0.5: 2, 1: 2, 3: 2, 4: 1, 2: 1})
{% endhighlight %}
</td>
<td>
{% highlight julia %}
Dict{Float64, Int64} with 5 entries:
4.0 => 1
2.0 => 1
0.5 => 2
3.0 => 2
1.0 => 2
{% endhighlight %}
</td>
</tr>
</table>


## Ternary operator
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
a, b = 3, 8
-1 if a > b else 0
{% endhighlight %}
</td>
<td>
{% highlight julia %}
a, b = 3, 8;
a < b ? -1 : 0
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
0
{% endhighlight %}
</td>
<td>
{% highlight julia %}
-1
{% endhighlight %}
</td>
</tr>
</table>


## Scoping
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
x = 10
def f():
    global x 
    x = 3
    return x
f()
{% endhighlight %}
</td>
<td>
{% highlight julia %}
x = 10
function f()
    global x
    x = 3
    return x
end
f()
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
3
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3
{% endhighlight %}
</td>
</tr>
</table>


## Iterables
### map
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
z = map(lambda x,y: x+y, 
    ['zero_','one_','two_'],
    ['1','2','3'])
list(z)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
map((x,y)-> x*y, 
    ["zero_","one_","two_"],
    ['1','2','3'])
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
['zero_1', 'one_2', 'two_3']
{% endhighlight %}
</td>
<td>
{% highlight julia %}
3-element Vector{String}:
 "zero_1"
 "one_2"
 "two_3"
{% endhighlight %}
</td>
</tr>
</table>


### zip
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
x,y,z = zip([1,2,3],[4,5,6],[8,9,10])
x
y
z
{% endhighlight %}
</td>
<td>
{% highlight julia %}
x,y,z = zip([1 2 3],[3 4 5],[8 9 10])
x
y
z
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
(1, 4, 8)
(2, 5, 9)
(3, 6, 10)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
(1, 3, 8)
(2, 4, 9)
(3, 5, 10)
{% endhighlight %}
</td>
</tr>
</table>


### enumerate
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
for (x,y) in enumerate(['a','b','c']):
    print(x,y)
{% endhighlight %}
</td>
<td>
{% highlight julia %}
for (x,y) in enumerate(['a' 'b' 'c'])
    println(x,y)
end
{% endhighlight %}
</td>
</tr>
<tr>
<td>Output</td>
<td>
{% highlight python %}
0 a
1 b
2 c
{% endhighlight %}
</td>
<td>
{% highlight julia %}
1a
2b
3c
{% endhighlight %}
</td>
</tr>
</table>

## url requests
<table>
<tr>
<th></th>
<th>Python</th>
<th>Julia</th>
</tr>
<tr>
<td>Code</td>
<td>
{% highlight python %}
import requests
#pings web page
requests.get("http://www.google.com")
#prints page html to output
requests.get("http://www.google.com").text
{% endhighlight %}
</td>
<td>
{% highlight julia %}
using Downloads
#pings web page
Downloads.request("www.google.com")
#prints page html to output
Downloads.download("www.google.com",stdout)
{% endhighlight %}
</td>
</tr>
</table>