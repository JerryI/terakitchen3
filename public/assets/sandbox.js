    core.FrontEndRemoveCell = function(args, env) {
      var input = JSON.parse(interpretate(args[0]));
      if(input["parent"] === "") {
        document.getElementById(input["id"]).remove();
      } else {
        document.getElementById(input["id"]+"---"+input["type"]).remove();
      }
    }

    core.FrontEndCreateCell = function(args, env) {
      var input = JSON.parse(interpretate(args[0]));
      console.log(input);

      
    
      var target;

      if (input["parent"] === "") {
        // create a new div element
        const newDiv = document.createElement("div");
        newDiv.id = input["id"];
        newDiv.classList.add("parent-node");
        document.getElementById("console").appendChild(newDiv);
        
        target = newDiv;
      } else {
        target = document.getElementById(input["parent"]);
      }

      var notebook = input["sign"];
      var uuid = input["id"];

      var newCell = CodeMirror(target, {value: input["data"], mode:  "mathematica", extraKeys: {
        "Shift-Enter": function(instance) { 
           eval(instance.getValue(), notebook, uuid);
        },
       }});

      newCell.on("blur",function(cm,change){ socket.send('CellObj["'+cm.display.wrapper.id.split('---')[0]+'"]["data"] = "'+cm.getValue()+'";'); });

      var wrapper = newCell.display.wrapper;

      wrapper.id = input["id"]+"---"+input["type"];

      wrapper.classList.add(input["type"] + '-node');

      last = input["id"];

    }
