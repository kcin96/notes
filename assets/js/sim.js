// Graph plot
function sandboxplot(x1,x2,v,autoscale=false){
    simbox = document.querySelector('[id^=simbox]')
    r = [-10, 10]    
    var trace1 = {
        x: x1,
        y: x2,
        mode: 'lines',
        type: 'scatter',
        name: 'f(x)'
    };
    v1 = v._data[0]
    v2 = v._data[1]
    console.log(v._data[1])
    var trace2 = {
        x: v1,
        y: v2,
        mode: 'markers+lines',
        marker: {
            color: 'red',
            symbol: 'circle'
    
        },
        type: 'scatter',
        name: 'vec'
    }

    var layout = {
        xaxis: {
            autosize: true,
            range: r
        },
        yaxis: {
            autosize: true, 
            range: r
        }
    };
    data = [trace1,trace2]
    if (autoscale === true){
        Plotly.newPlot('graphplot',data)
    }
    else{
        Plotly.newPlot('graphplot',data,layout)
    }
}

// UI input 
function user_form(){
    xscale = document.querySelector("#xscale")
    yscale = document.querySelector("#yscale")
    xval = document.querySelector("#xscaleval")
    yval = document.querySelector("#yscaleval")
    autoscale_graph = document.querySelector("#autoscale_graph")

    xscale.addEventListener('change', ()=>{
        xval.innerHTML = xscale.value
        compute_and_plot(xscale.value,yscale.value,autoscale_graph.checked)
    });

    yscale.addEventListener('change', ()=>{
        yval.innerHTML = yscale.value
        compute_and_plot(xscale.value,yscale.value,autoscale_graph.checked)
    });

    $('#autoscale_graph').on('change.bootstrapSwitch', function(e) {
        console.log(e.target.checked);
        compute_and_plot(xscale.value,yscale.value,e.target.checked)
    });

}

// computes scaling and updates graph
function compute_and_plot(x=1,y=1,autoscale=false){
    var x1 = math.divide(range(201,-100),100)
    var x2 = math.map(math.subtract(1,math.map(x1,math.square)),math.sqrt)
    x1 = x1.concat(math.subtract(0,x1))
    x2 = x2.concat(math.subtract(0,x2))

    // arrow vectors
    v1 = [[-1],[0]]
    dv = [[0.5],[0.5]]
    dv2 = [[0.5],[-0.5]]

    // Transformation matrix
    T = math.matrix([[parseInt(x),0],[0,parseInt(y)]])

    // scale step
    x1 = math.multiply(x1,parseInt(x))
    x2 = math.multiply(x2,parseInt(y))

    // start point
    v1 = math.multiply(T,v1)
    // end point 1
    v2 = math.add(v1,math.multiply(T,dv))
    // end point 2
    v3 = math.add(v1,math.multiply(T,dv2))
    
    // 2xn matrix
    v = math.concat(v1,v2)
    v = math.concat(v,v1)
    v = math.concat(v,v3)

    // update graph plot
    sandboxplot(x1,x2,v,autoscale)
}

// range function
function range(size,start=0){
    return [...Array(size).keys()].map(i=>i+start)
}

document.addEventListener("DOMContentLoaded", () => {
    user_form()
    compute_and_plot(1,1)

});