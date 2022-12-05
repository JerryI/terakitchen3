(*binging to the events*)
Options[NotebookCreate] = {
    "name" -> "Untitled",
    "id" :> CreateUUID[],
    "evaluator" -> SimpleEvaluator
};

NotebookCreate[OptionsPattern[]] := (
    With[{id = OptionValue["id"]},

        notebooks[id] = <|
            "name" -> OptionValue["name"],
            "evaluator" -> OptionValue["evaluator"]
        |>;

        notebooks[id, "cell"] = CellObj["sign"->id, "type"->"input", "data"->"\"Hello World!\""];
        id
    ]
);

(*access only via websockets*)
NotebookOpen[id_] := (
    Block[{fireEvent = CellEventFire[client]},
        CellObjGenerateTree[notebooks[id, "cell"]];
    ];
);

NotebookCellEvaluate[id_, cellid_] := (
    Block[{fireEvent = CellEventFire[client]},
        CellObjEvaluate[CellObj[cellid], notebooks[id, "evaluator"] ];
    ];
);

CellEventFire[addr_]["NewCell"][cell_] := (
    (*actually frirstly you need to check!*)
    With[
        {
            obj = <|
                        "id"->cell[[1]], 
                        "sign"->cell["sign"],
                        "type"->cell["type"],
                        "data"->cell["data"],
                        "child"->If[NullQ[ cell["child"] ], "", cell["child"][[1]]],
                        "parent"->If[NullQ[ cell["parent"] ], "", cell["parent"][[1]]],
                        "next"->If[NullQ[ cell["next"] ], "", cell["next"][[1]]],
                        "prev"->If[NullQ[ cell["prev"] ], "", cell["prev"][[1]]]
                    |>
        },

        WebSocketPublish[server, FrontEndCreateCell[ExportString[obj, "JSON"] ], addr];
    ];
);