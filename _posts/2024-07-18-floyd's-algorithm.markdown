---
layout: sidebar
title:  "Floyd's cycle finding algorithm"
date:   2024-04-08 21:16:00 +0100
categories: Algorithm
---

* toc
{:toc}
<script src="{{ "assets/js/floyd_cycle.js" | relative_url }}" type="text/javascript"></script>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js" charset="utf-8"></script>
<script src="https://unpkg.com/mathjs/lib/browser/math.js"></script>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>


## Introduction
Floyd's cycle finding algorithm or tortoise and hare algorithm finds cycles in a linked list. If there is a cycle in a linked list, a slow pointer can meet up with the fast pointer. An analogy would be that the slow tortoise has "caught up" with the fast hare. 

## Linked list with a cycle
<svg width="500" height="170">

<circle cx="50" cy="60" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="100" cy="60" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="150" cy="60" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="200" cy="20" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="250" cy="60" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="225" cy="110" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<circle cx="175" cy="110" r="10" stroke="rgb(0,0,255)" stroke-width="2" fill="white"></circle>
<polyline points="60,60 90,60 80,55 90,60 80,65" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="110,60 140,60 130,55 140,60 130,65" style="fill:none;stroke:black;stroke-width:1.5"/>
<path d="M 150 50 Q 160 20 190 20" style="fill:none;stroke:black;stroke-width:1.5"/>
<path d="M 210 20 Q 250 20 250 50" style="fill:none;stroke:black;stroke-width:1.5"/>
<path d="M 250 70 Q 260 90 235 110" style="fill:none;stroke:black;stroke-width:1.5"/>
<path d="M 215 115 Q 200 120 185 115" style="fill:none;stroke:black;stroke-width:1.5"/>
<path d="M 165 105 Q 150 90 150 70" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="190,20 180,15 190,20 180,25" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="250,50 245,40 250,50 255,40" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="235,110 245,110 235,110, 240,100" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="185,115 195,110 185,115, 195,120" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="150,70 155,80 150,70, 145,80" style="fill:none;stroke:black;stroke-width:1.5"/>
<polyline points="30,100 40,80 40,120, 40,80 50,100" style="fill:none;stroke:green;stroke-width:1.5"/>
<polyline points="50,100 60,80 60,120, 60,80 70,100" style="fill:none;stroke:red;stroke-width:1.5"/>

<text x="30" y="130" font-size='15' style="stroke:green">p1</text>
<text x="60" y="130" font-size='15' style="stroke:red">p2</text>
<text x="20" y="150" font-size='15' style="stroke:green">slow</text>
<text x="60" y="150" font-size='15' style="stroke:red">fast</text>
</svg>

Let $a$ be the distance from the start node to the cycle start node, $b$ the distance from the cycle start node to the meet node and $T$ be the cycle length.  

Slow pointer $p_1$ moves 1 node at each time step.
Fast pointer $p_2$ moves 2 nodes at each time step. 

1. If there is no cycle, $p_2$ will reach the end(last) node first.
2. If there is a cycle, both pointers meet at a node within the cycle.

## Proof
$$\begin{aligned}
Distance(p_1) &= a + b + k_1T\\
Distance(p_2) &= a + b + k_2T\\
Distance(p_2) &= 2Distance(p_1)\\
a + b + k_2T &= 2(a + b + k_1T)\\
a + b &= (k_2-2k_1)T\\
a &= (k_2-2k_1)T - b\\
a &= kT - b\\
&where \ k_1<k_2, k_1, k_2, k  \in \mathbb{Z}^{+}\\
\end{aligned}$$
Since $k$ is an non-negative integer, both pointers meet at a node within the cycle. Hence
<div class="alert alert-secondary" role="alert">
If fast and slow pointers meet $\Rightarrow$ there is a cycle.
</div>

Note that when both pointers meet, the fast pointer is $b$ steps ahead of the cycle start node. Hence, if the fast pointer advances $a$ steps forward, it will be at the cycle start node. If we reset the slow pointer to the start node, and advance both pointers 1 step at a time, both pointers will meet at the cycle start node.

## Modulo arithmetic
Position of fast pointer (with respect to cycle index) is essentially
$$\begin{aligned}
(D-a) \ mod \ T\\
\end{aligned}$$
Position of fast pointer (with respect to node index) is essentially
$$\begin{aligned}
a + (D-a) \ mod \ T\\
\end{aligned}$$
where $D$ is the distance from start node, $a$ the distance from start node to the cycle start node, $T$ is the cycle period.

## Code implementation
<nav>
  <div class="nav nav-tabs" id="nav-tab" role="tablist">
    <button class="nav-link active" id="julia-tab1" data-bs-toggle="tab" data-bs-target="#julia1" type="button" role="tab" >Julia</button>
    <button class="nav-link" id="cpp-tab2" data-bs-toggle="tab" data-bs-target="#cpp1" type="button" role="tab" >C++</button>
  </div>
</nav>
<div class="tab-content" id="nav-tabContent">
  <div class="tab-pane fade show active" id="julia1" role="tabpanel" >

{%highlight julia%}
mutable struct Node
    index::Int32
    next_node::Union{Node, Nothing}
end

# linked list
linked_list = Node(0, Node(1, Node(2, Node(3, nothing))))

# cycle list
cycle_start = Node(2, nothing)
cycle = Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, cycle_start)))))))
cycle_start.next_node = cycle
cycle_list =  Node(0, Node(1, cycle_start))

# floyd cycle finding algorithm
function floyd(start_node)
    p1 = start_node   # slow pointer
    p2 = p1   # fast pointer

    while true
        # checks if fast pointer reachs end of linked list
        if p2.next_node == nothing || p2.next_node.next_node == nothing
            println("No cycle")
            break
        end

        # advance fast and slow pointer
        p1 = p1.next_node
        p2 = p2.next_node.next_node
        println(p1.index,", ",p2.index)

        # if pointers meet, found meeting node
        if p1 == p2
            println("Meet node: ", p1.index)

            # find start of cycle
            p1 = cycle_list
            while (p1 != p2)
                p1 = p1.next_node
                p2 = p2.next_node
            end
            println("Cycle start node: ", p1.index)
            break
        end
    end
end
{%endhighlight%}

{%highlight julia%}
floyd(linked_list)
# Output:
#=
1, 2
No cycle 
=#
{%endhighlight%}

{%highlight julia%}
floyd(cycle_list)
# Output:
#=
1, 2
2, 4
3, 6
4, 8
5, 2
6, 4
7, 6
8, 8
Meet node: 8
Cycle start node: 2
=#
{%endhighlight%}
  </div>
  <div class="tab-pane fade" id="cpp1" role="tabpanel">
{%highlight cpp%}
#include <iostream>
using namespace std;

struct Node{
    int index;
    Node* next;
};

void build_linked_list(Node* head, int n_nodes);
void build_cycle_list(Node* head, int a, int T);
void print_list(Node* node);
void floyd(Node* head);

int main(){
    // linked list
    Node* ll_head = new Node();
    ll_head -> index = 0;
    ll_head -> next = NULL;
    build_linked_list(ll_head, 10);
    cout<<"linked list:";
    print_list(ll_head);
    floyd(ll_head);
    cout<<endl;

    //cycle list
    Node* cycle_head = new Node();
        cycle_head -> index = 0;
        cycle_head -> next = NULL;
    build_cycle_list(cycle_head, 3, 6);
    floyd(cycle_head);
    return 0;
}

void build_linked_list(Node* head, int n_nodes){
    Node* ptr = head;
    for (int i=1; i<=n_nodes; i++){
        Node* node = new Node();
            node -> index = i;
            node -> next = NULL;
        ptr -> next = node;
        ptr = node;
    }
}

void build_cycle_list(Node* head, int a, int T){
    Node* ptr = head;
    for (int i=1; i<a+T; i++){
        Node* node = new Node();
            node -> index = i;
            node -> next = NULL;
        ptr -> next = node;
        ptr = node;
    }
    Node* ptr2 = head;
    for (int i=0; i<a; i++){
        ptr2 = ptr2 -> next;
    }
    ptr -> next = ptr2;
}

void print_list(Node* node){
    while (node -> next!=NULL){
        cout<<node -> index;
        node = node -> next;
    }
    cout<<endl;
}

void floyd(Node* head){
    Node* p1 = head; //slow pointer
    Node* p2 = p1;   //fast pointer
    while (p2 -> next != NULL){
        if (p2 -> next -> next != NULL){
            p1 = p1 -> next;
            p2 = p2 -> next -> next;
            cout<<p1 -> index<<","<<p2 -> index<<endl;
            if (p1 == p2){
                cout<<"Meet node: "<<p1 -> index<<endl;
                //find start of cycle
                p1 = head;
                while (p1 != p2){
                    p1 = p1 -> next;
                    p2 = p2 -> next;
                }
                cout<<"Cycle start node: "<<p1 -> index<<endl;
                return;
            }
        }
    }
    cout<<"no cycle"<<endl;
}
{%endhighlight%}

{%highlight cpp%}
// Output
linked list:0123456789
1,2
2,4
3,6
4,8
5,10
no cycle

1,2
2,4
3,6
4,8
5,4
6,6
Meet node: 6
Cycle start node: 3
{%endhighlight%}
</div>
</div>

## Interactive Demo
<div class="card" style="height:500px; width: 900px;" id="simbox">
    <div class="container text-center" style="padding:10px">
        <div class="row">
            <div class="col-3">
                <div class="form-control form-control-sm" style="height:450px; width: 200px;" >
                    <label for="customRangex" class="form-label">a (Distance from the start node to the cycle start node) range</label>
                    <p id="xscaleval">1</p>
                    <input type="range" class="form-range" min="1" max="5" id="xscale" value=1>
                    <label for="customRangey" class="form-label">T (cycle period) range</label>
                    <p id="yscaleval">1</p>
                    <input type="range" class="form-range" min="1" max="10" id="yscale" value=1>
                </div>
            </div>
            <div class="col-6" id="nodeplot">
            </div>
        </div>
    </div>

</div>

## References
1. [https://cp-algorithms.com/others/tortoise_and_hare.html](https://cp-algorithms.com/others/tortoise_and_hare.html)

