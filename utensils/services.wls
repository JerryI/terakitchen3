CallService[args_]:=
  Module[{unit = Get[StringTemplate["services/``/manifest"][args["service"]]], err},

      console["log", ">>> call service with args: ``", args//Compress];

      console["log","apply `` for", args["service"]];
      console["log","--- ``", #]&/@args["list"];

      err = Catch[

        With[
          {
              newset = If[KeyExistsQ[args, "params"],
                          ValidateService[unit["parameters"], args]
                      ,
                          unit["parameters"]
                      ]
          },

          console["log", "-- newset parameters: ``", newset//Compress];


          ProcessStart[CreateUUID[], "name"->args["service"], "task"->StringTemplate["services/``/``.wls"][args ["service"], args["service"]], "init"->newset, "list"->args["list"], "after"->args["after"]];
        ];
      ];

      (*check errors*)
      If[StringQ[err],
        PushNotification["utensils/services/callservice", StringTemplate["An error occured. &#129488; Check the field <i> ``</i>"][err]];
      ];
  ];

ValidateService[parameters_, args_] := 
  (
  Function[s,
    If[MissingQ[args["params", s["symbol"]]], 
      <|"symbol"->s["symbol"], "value"->s["value"]|>
    ,
      With[{val = Switch[s["type"], 
                            "text", 
                              If[StringQ[#], #, Throw[StringTemplate["Provided data `` is not a string. Which is not valid for the selected input field."][#]]]&@args["params", s["symbol"]],
                            "integer",
                              If[IntegerQ[#], #, Throw[StringTemplate["Provided data `` is not a integer. Which is not valid for the selected input field."][#]]]&@ToExpression[args["params", s["symbol"]]],
                            "symbol",
                              If[NumberQ[#//N], #, Throw[StringTemplate["Provided data `` is not a prober computable number. Which is not valid for the selected input field."][#]]]&@ToExpression[args["params", s["symbol"]]],
                            "real",
                              If[NumberQ[#], #, Throw[StringTemplate["Provided data `` is not a number. Which is not valid for the selected input field."][#]]]&@ToExpression[args["params", s["symbol"]]],
                            "bool",
                              If[BooleanQ[#], #, Throw[StringTemplate["Provided data `` is not a boolean type. Which is not valid for the selected input field."][#]]]&@If[args["params", s["symbol"]] === "on", True, False]                  
                         ]
          },

          <|"symbol"->s["symbol"], "value"->val|>
      
      ]
    ]
    

  ]/@parameters
);  

ProcessFiles[args_] := Module[{descid = CreateUUID[], donelist, pipeline, services, item, diff, unit, pipe, firstpipe,firstservice, service, deps, sequence, err},

  PushNotification["utensils/services/processfiles", "Started to process "<>ToString[args["files"]//Length]<>" files"];

  (*fetching for the first service*)
    
  services = Association[(#->Get["services/"<>#<>"/manifest"])&/@FileNameTake/@FileNames[All, "services"] ];

  (*process all files*)
  diff = args["files"];

  err = Catch[
   
    If[Not[(ThicknessNormalise["mm"][args["thickness"] ])//NumberQ//TrueQ],  Throw[StringTemplate["`` is not a valid thickness value"][args["thickness"] ] ] ];


    diff = Table[ImportFile[i, args], {i, diff}];

    console["log", ">> imported ``", #]/@diff;

    donelist={};
    sequence = {};

    Do[
      unit = services[service];

      If[SubsetQ[donelist, unit["dependencies"] ] == False && args["services", service, "allowed"] === "on",
        console["log", "an error was occured. Dependencies are not matched!"];
        deps = "";

        (deps = deps <> # <> ",")&/@unit["dependencies"];

        deps = StringDrop[deps,-1];

        PushNotification["utensils/services/processfiles", "<span class=\"badge badge-danger\">Dependencies are not matched</span> for "<>service<>"."<>"<br>Need: "<>deps<>" Skipping..."];
        Continue[];
      ];

      (*virtual env*)
      If[args["services", service, "allowed"] === "on",
          AppendTo[donelist, service];
          sequence = {sequence, <|"params" -> args["services", service], "list" -> diff, "service" -> service|>};
      ];

      console["log", "service >> name ``", service];

    , {service, SortBy[services//Keys, services[#, "priority"]&]}];

    (*add to the folder*)
    If[args["folder"] != "Workbook",
      folders[args["folder"], "list"] = Flatten[{folders[args["folder"], "list"], diff}];
    ];

    sequence = sequence // Flatten;


    With[{e = args["exp"]},
      Module[
        {
          parent = Hold[
            experiments[e, "date"] = Now;
            PushNotification["utensils/services/processfiles", "Finished. Click here <br>"<>"<a class=\"btn btn-outline-info\"  href=\"http://"<>$env["addr"]<>"/experiment/index.wsp?id="<>e<>"\">open</a>"];
            (*PushNotification["Server", "Backing up the database in 7 minutes..."];
            SessionSubmit[ ScheduledTask[CBackup, {Quantity[7, "Minutes"]}, AutoRemove->True] ];*)
          ]
        },

        With[{a = Join[#, <|"after"->parent|>]},

          parent = Hold[SessionSubmit[CallService[a]] ];

        ]&/@Reverse[sequence];

        parent//ReleaseHold;
      ];
    ];
  ];

  (*checking for errors*)
  If[StringQ[err],
    PushNotification["Import", StringTemplate["<span class=\"badge badge-danger\">An error has occured</span> &#128511; The evaluation was stopped <i>``</i>. Try again, please."][err] ];
  ];
];

ImportFile[file_, args_] := Module[{input, fields, TransferField, TransferTheRest, exp=args["exp"], record = Association[]}, 
  
   console["log", "pipe >> importing..."]; 
   fields = args["parser"];

   TransferField[i_, string_, fields_] := Module[{newitem = i, newstring = string}, 
      If[Length[fields] == 0, Return[{i, string}]]; 
      newitem[fields[[1,1]]] = fields[[1,3]][First[StringCases[newstring, RegularExpression[fields[[1,2]]] -> "$1"]]]; 

      If[newitem[fields[[1,1]]] === $Failed, Throw[StringTemplate["There is a problem with a field ``. Check you parser settings in the import modal window. Function used ``"][fields[[1,1]], ToString[fields[[1,3]], InputForm]]]];
      
      newstring = StringReplace[newstring, StringCases[newstring, RegularExpression[fields[[1,2]]]] -> ""]; 
      
      TransferField[newitem, newstring, Drop[fields, 1]]
   ];

   TransferTheRest[i_, string_] := Module[{newitem = i, newstring = string, rest}, 
      rest = StringCases[StringReplace[newstring, ".csv" -> ""], RegularExpression["_?[^_]+_?"]]; 
      ((newitem[StringReplace[#, "_" -> ""]] = True) & ) /@ rest; 
      
      newitem
   ]; 


   With[{}, 
      console["log", file["filename"]];

      record["id"] = StringDelete[exp, RegularExpression["(\d|\.|\s.*)"]]<>"-"<>StringTake[CreateUUID[], 4];
     
      If[KeyExistsQ[args, "thickness"],  record["thickness"] = args["thickness"]];
      If[KeyExistsQ[args, "customtag"],  
        With[{assoc = Association[ToExpression["{"<>args["customtag"]<>"}"] // Flatten ]},
          If[!AssociationQ[assoc], Throw[StringTemplate["tag `` is invalid."][args["customtag"]]]]; 
          record = Join[record, assoc];        
        ]
      ];
     
      record["experiment"] = exp; 
      record["attributes"] = <|"usable-range"->Sequence[All]|>;
     
      If[KeyExistsQ[args, "descid"], record["description"] = args["descid"]];
      
      If[args["manual"] === True,
        record["$manual"] = True;
      ,
        record = (TransferTheRest @@ TransferField[record, FileBaseName[file["filename"]], fields])//Quiet; 
      ];
      
         
      Module[{tds = Drop[ ImportString[StringReplace[ByteArrayToString[file["data"]], {";"->",", "E"->"*10^"}], "CSV"],  1]},
        If[!ArrayQ[tds], Throw["Imported CSV file is not array"]];
        If[!NumberQ[tds[[1,1]]], Throw[StringTemplate["imported cell `` is not a number"][tds[[1,1]]]]];

        If[Length[First@tds] > 2,
          PushNotification["utensils/services/importfiles", StringTemplate["An extra `` columns were found. For the TD signal we will ignore them."][Length[tds[[2]]] - 2] ];
          tds = Take[#,2]&/@tds;     
        ];

        If[Length[First@tds] < 2, Throw["The number of columns is less than two."]];

        record["data"] = <|"tds" ->  tds|>; 
      ];


      record["filename"] = file["filename"] // FileBaseName;
      record["date"] = Now; 

      collection[record["id"]] = .;

      CPut[record]; 
     
   ];

   record["id"]
]