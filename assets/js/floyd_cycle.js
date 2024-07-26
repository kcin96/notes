// nodes plot
function sandboxplot(x,y,a){
    simbox = document.querySelector('[id^=simbox]') 
    var marker_prop = {
        color: 'aqua',
        symbol: 'circle',
        size: 30,
        opacity: 1,
        line: {
            color:'blue',
            width:4
        }
    }

    var trace = {
        x: x,
        y: y,
        text: range(x.length,0),
        mode: 'markers+lines+text',
        marker: marker_prop
    }

    var end_connection = {
        x: [x[a], x[x.length-1]],
        y: [y[a], y[y.length-1]],
        mode: 'lines',
        marker: marker_prop
    }

    var layout = {
        showlegend: false,
        annotations: [
            {
                x: 1,
                y: 3.5,
                text:'slow',
                showarrow: true,
                arrowhead: 3,
                arrowcolor: 'green',
                font: {
                    family: 'Courier New',
                    color: 'green'
                  },
            },
            {
                x: 2,
                y: 4.5,
                text:'fast',
                showarrow: true,
                arrowhead: 3,
                arrowcolor: 'red',
                font: {
                    family: 'Courier New, monospace',
                    color: 'red'
                  },
            }
        ],
        xaxis:{
            showline: false,
            zeroline: false,
            showgrid: false
        },
        yaxis:{
            showline: false,
            zeroline: false,
            showgrid: false
        }

    }
    data = [trace, end_connection]
    Plotly.newPlot('nodeplot',data,layout)
}

// UI input 
function user_form(){
    xscale = document.querySelector("#xscale")
    yscale = document.querySelector("#yscale")
    xval = document.querySelector("#xscaleval")
    yval = document.querySelector("#yscaleval")

    xscale.addEventListener('change', ()=>{
        //a
        xval.innerHTML = xscale.value
        compute_and_plot(xscale.value,yscale.value)
    });

    yscale.addEventListener('change', ()=>{
        //T
        yval.innerHTML = yscale.value
        compute_and_plot(xscale.value,yscale.value)
    });

}

// computes node graph and display
function compute_and_plot(a=1,period=1){
    var linear_list = range(parseInt(a),0)
    var cycle_list = range(parseInt(period),parseInt(a))
    var x = []
    var y = []

    linear_list.forEach(element => {
        x.push(element*5)
        y.push(0)
    });

    var r = a*5/2
    var startx_pos = x[x.length-1]+2*r
    var starty_pos = y[y.length-1]
    cycle_list.forEach(index => {
        if (parseInt(period)%2==0){    
            angle = 2*3.142/period*(index-a)  //index-a subtracts bias a
        }
        else{
            angle = 2*3.142/period*(index-a+1)
        }

        // modify angle to get clockwise behaviour
        if (angle<3.142){
            angle = 3.142-angle
        }
        else{
            angle = -(angle-3.142)
        }

        x.push(startx_pos+r*math.cos(angle))
        y.push(starty_pos+r*math.sin(angle))
    });

    // update nodes plot
    sandboxplot(x,y,a)
    range(parseInt(a)+parseInt(period), 0).forEach( i => { 
        update(i, parseInt(a), parseInt(period), x, y)
    });

}

// range function
function range(size,start=0){
    return [...Array(size).keys()].map(i=>i+start)
}

//function step_foward
// Args:=index: step number, T: cycle period  
function step_forward(index, a, T){
    slow = a + (index - a) % T
    fast = a + (2 * index - a) % T 
    return [slow, fast]
}

//function updates and animates slow and fast pointer 
function update(index, a, T, x, y){
    var [slow, fast] = step_forward(index, a, T);
    if (slow == fast){
        console.log(index)
    }

    var layout = {
        showlegend: false,
        annotations: [
            {
                x: x[slow],
                y: y[slow],
                text:'slow',
                showarrow: true,
                arrowhead: 3,
                arrowcolor: 'green',
                font: {
                    family: 'Courier New',
                    color: 'green'
                    },
            },
            {
                x: x[fast],
                y: y[fast],
                text:'fast',
                showarrow: true,
                arrowhead: 3,
                arrowcolor: 'red',
                font: {
                    family: 'Courier New, monospace',
                    color: 'red'
                    },
            }
        ],
        xaxis:{
            showline: false,
            zeroline: false,
            showgrid: false,
            autosize: true,
            
        },
        yaxis:{
            showline: false,
            zeroline: false,
            showgrid: false,
            autosize: true,
        }

    }
    Plotly.animate('nodeplot',{  
        layout: layout      
    }, {
        transition:{
            duration: 0
        },
        frame: {
            duration: 1000,
            redraw: false
        },
    });

}

document.addEventListener("DOMContentLoaded", () => {
    compute_and_plot(1,1)
    user_form()
});