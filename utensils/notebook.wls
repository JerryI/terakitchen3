(*binging to the events*)
Unprotect[NotebookCreate];
Unprotect[NotebookOpen];
Unprotect[NotebookEvaluate];

ClearAll[NotebookOpen];
ClearAll[NotebookCreate];
ClearAll[NotebookEvaluate];

Options[NotebookCreate] = {
    "name" -> "Untitled",
    "id" :> CreateUUID[],
    "evaluator" -> SimpleEvaluator,
    "ref" -> Null,
    "sandboxed" -> False,
    "data" -> "1+1",
    "pre" -> Null
};

NotebookCreate[OptionsPattern[]] := (
    With[{id = OptionValue["id"]},

        notebooks[id] = <|
            "name" -> OptionValue["name"],
            "evaluator" -> OptionValue["evaluator"],
            "ref" -> OptionValue["ref"],
            "sandboxed" -> OptionValue["sandboxed"],
            "pre" -> OptionValue["pre"]
        |>;

        notebooks[id, "cell"] = CellObj["sign"->id, "type"->"input", "data"->OptionValue["data"]];
        id
    ]
);

NotebookRemove[id_] := (
    CellObjRemoveAllNext[notebooks[id, "cell"] ];
    CellObjRemove[notebooks[id, "cell"] ];

    Switch[ToString[notebooks[id, "ref"]["type"] ],
        "experiment",
            experiments[notebooks[id, "ref", "id"], "notebooks"] = Complement[notebooks[id, "ref", "id"], id];
        ,
        "folder",
            folders[notebooks[id, "ref", "id"], "notebook"] = .;
        ,
        "item",
            collection[notebooks[id, "ref", "id"], "notebook"] = .;
    ];
    
    notebooks[id] = .;
);

(*access only via websockets*)
NotebookAttach[id_, proc_, init_:Null] := Module[{pid = proc},
    If[proc === "master", notebooks[id]["evaluator"] = SimpleEvaluator; PushNotification["utensils/notebook", "<span class=\"badge badge-danger\">Master kernel attached</span>"]; Return["master", Module] ];
    If[proc === "Null"  ,
        pid = CreateUUID[];
        ProcessStart[pid, "name" -> "sandbox", "task" -> "svcore/sandbox.wls", "pre" -> notebooks[id]["pre"] ];
        WebSocketSend[client, FrontEndAddKernel[pid, "sandbox"] ];
    ,
        JTPClientSend[settings["processes", proc, "listener"], Import["svcore/sandbox.wls"]; ];
    ];

    

    notebooks[id]["evaluator"] = RemoteEvaluator[pid];
];

NotebookOpen[id_] := (
    Block[{fireEvent = CellEventFire[client]},
        CellObjGenerateTree[notebooks[id, "cell"]];
    ];
);

NotebookEvaluate[id_, cellid_] := (
    Block[{fireEvent = CellEventFire[client]},
        CellObjEvaluate[CellObj[cellid], notebooks[id, "evaluator"] ];
    ];
);

NotebookOperate[cellid_, op_] := (
    Block[{fireEvent = CellEventFire[client]},
        op[CellObj[cellid]];
    ];
);

CellEventFire[addr_]["NewCell"][cell_] := (
    (*looks ugly actually. we do not need so much info*)
    With[
        {
            obj = <|
                        "id"->cell[[1]], 
                        "sign"->cell["sign"],
                        "type"->cell["type"],
                        "data"->If[cell["data"]//NullQ, "", ExportString[cell["data"], "String", CharacterEncoding -> "UTF8"] ],
                        "child"->If[NullQ[ cell["child"] ], "", cell["child"][[1]]],
                        "parent"->If[NullQ[ cell["parent"] ], "", cell["parent"][[1]]],
                        "next"->If[NullQ[ cell["next"] ], "", cell["next"][[1]]],
                        "prev"->If[NullQ[ cell["prev"] ], "", cell["prev"][[1]]],
                        "storage"->cell["storage"]
                    |>,
            
            template = LoadPage["public/assets/cells/"<>cell["type"]<>".wsp", {id = cell[[1]]}]
        },

        WebSocketSend[addr, FrontEndCreateCell[template, ExportString[obj, "JSON"] ]];
    ];
);

CellEventFire[addr_]["RemovedCell"][cell_] := (
    (*actually frirstly you need to check!*)
  
    With[
        {
            obj = <|
                        "id"->cell[[1]], 
                        "sign"->cell["sign"],
                        "type"->cell["type"],
                        "child"->If[NullQ[ cell["child"] ], "", cell["child"][[1]]],
                        "parent"->If[NullQ[ cell["parent"] ], "", cell["parent"][[1]]],
                        "next"->If[NullQ[ cell["next"] ], "", cell["next"][[1]]],
                        "prev"->If[NullQ[ cell["prev"] ], "", cell["prev"][[1]]]
                    |>
        },

        WebSocketSend[addr, FrontEndRemoveCell[ExportString[obj, "JSON"] ]];
    ];
);

CellEventFire[addr_]["ClearStorage"][cell_] := (
    (*actually frirstly you need to check!*)
  
    With[
        {
            obj = <|
                        "id"->cell[[1]], 
                        "storage"->Keys[cell["storage"]]
                    |>
        },

        WebSocketSend[addr, FrontEndClearStorage[ExportString[obj, "JSON"] ]];
    ];
);

CellEventFire[addr_]["CellError"][cell_, text_] := WebSocketSend[addr, FrontEndCellError[cell[[1]], text]];

CellEventFire[addr_]["CellMove"][cell_, parent_] := (
    With[
        {   template = LoadPage["public/assets/cells/input.wsp", {id = cell[[1]]}],
            obj = <|
                    "cell"-> <|
                        "id"->cell[[1]], 
                        "sign"->cell["sign"],
                        "type"->cell["type"],
                        "child"->If[NullQ[ cell["child"] ], "", cell["child"][[1]]],
                        "parent"->If[NullQ[ cell["parent"] ], "", cell["parent"][[1]]],
                        "next"->If[NullQ[ cell["next"] ], "", cell["next"][[1]]],
                        "prev"->If[NullQ[ cell["prev"] ], "", cell["prev"][[1]]]
                    |>,

                    "parent"-> <|
                        "id"->parent[[1]], 
                        "sign"->parent["sign"],
                        "type"->parent["type"],
                        "child"->If[NullQ[ parent["child"] ], "", parent["child"][[1]]],
                        "parent"->If[NullQ[ parent["parent"] ], "", parent["parent"][[1]]],
                        "next"->If[NullQ[ parent["next"] ], "", parent["next"][[1]]],
                        "prev"->If[NullQ[ parent["prev"] ], "", parent["prev"][[1]]]                        
                    |>
                |>
        },

        WebSocketSend[addr, FrontEndMoveCell[template, ExportString[obj, "JSON"] ]];
    ];
);

CellEventFire[addr_]["CellMorph"][cell_] := (
    With[
        {
            obj = <|
                    "id"->cell[[1]], 
                    "sign"->cell["sign"],
                    "type"->cell["type"],
                    "child"->If[NullQ[ cell["child"] ], "", cell["child"][[1]]],
                    "parent"->If[NullQ[ cell["parent"] ], "", cell["parent"][[1]]],
                    "next"->If[NullQ[ cell["next"] ], "", cell["next"][[1]]],
                    "prev"->If[NullQ[ cell["prev"] ], "", cell["prev"][[1]]]
                |>
        },

        WebSocketSend[addr, FrontEndMorphCell[ExportString[obj, "JSON"] ]];
    ];
);

