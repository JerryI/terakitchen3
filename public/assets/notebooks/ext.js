//extensions

//2D Plot using Ploty.js
core.WListPloty = function(args, env) {
    const arr = JSON.parse(interpretate(args[0]));
    console.log("Ploty.js");
    console.log(arr);
    let newarr = [];
    arr.forEach(element => {
        newarr.push({x: element[0], y: element[1]});
    });
    Plotly.newPlot(env.element, newarr, {autosize: false, width: 500, height: 300, margin: {
        l: 30,
        r: 30,
        b: 30,
        t: 30,
        pad: 4
      }});
}