Options[CellObj] = {
    "type" -> "input",
    "child" -> Null,
    "parent" -> Null,
    "next" -> Null,
    "prev" -> Null,
    "display" -> Null,
    "data" -> Null
};

CellObj[OptionsPattern[]] := With[{cell = CreateUUID[]}, 

	CellObj[cell]["type"   ] = OptionValue["type"];
    CellObj[cell]["child"  ] = OptionValue["child"];
    CellObj[cell]["parent" ] = OptionValue["parent"];
    CellObj[cell]["next"   ] = OptionValue["next"];
    CellObj[cell]["prev"   ] = OptionValue["prev"];
    CellObj[cell]["display"] = OptionValue["display"];
    CellObj[cell]["data"]    = OptionValue["data"];

    CellObj[cell]
];

CellObj /: 
CellObjFindLast[CellObj[cell_]] := (
    Module[{next = CellObj[cell]},
        While[next["next"] =!= Null, next = next["next"] ];
        next
    ]   
);

CellObj /: 
CellObjFindFirst[CellObj[cell_]] := (
    Module[{prev = CellObj[cell]},
        While[prev["prev"] =!= Null, prev = prev["prev"] ];
        next
    ]   
);

CellObj /: 
CellObjFindParent[CellObj[cell_]] := (  
    Module[{next = CellObj[cell]},
        If[cell["parent"] === Null,
            While[next["prev"] =!= Null, next = next["prev"] ];   
        ];
        next["parent"]
    ]  
);

CellObj /: 
CellObjCreateChild[CellObj[cell_]] := (  
    Module[{next = CellObj[cell]["child"], new = CellObj[]},
        If[next =!= Null,
            While[next["next"] =!= Null, next = next["next"] ];
            next["next"] = new;
            new["prev"] = next;        
        ,
            CellObj[cell]["child"] = new;
            new["parent"] = CellObj[cell];
        ];
        new
    ]  
);

CellObj /: 
CellObjCreateNext[CellObj[cell_]] := (  
    Module[{next = CellObj[cell], new = CellObj[]},
        While[next["next"] =!= Null, next = next["next"]];
        next["next"] = new;
        new["prev"] = next;
        new
    ]  
);

CellObj /:
CellObjRemove[CellObj[cell_]] := ( 
    CellEventFire["RemovedCell"][CellObj[cell_]];

    Unset[CellObj[cell]["data"]];
    Unset[CellObj[cell]["type"]];
    Unset[CellObj[cell]["next"]];
    Unset[CellObj[cell]["prev"]];
    Unset[CellObj[cell]["parent"]];

    If[CellObj[cell]["child"] =!= Null,
        CellObjRemoveAllNext[CellObj[cell]["child"]];
        CellObjRemove[CellObj[cell]["child"]];
    ];

    Unset[CellObj[cell]["child"]];
    Unset[CellObj[cell]["display"]];
);

CellObj /:
CellObjRemoveAllNext[CellObj[cell_]] := ( 
    If[CellObj[cell]["next"] =!= Null, 
        Module[{next = CellObj[cell]["next"]},
            While[next["next"] =!= Null, next = next["next"]];
            While[next =!= CellObj[cell], next = next["prev"]; CellObjRemove[next["next"]]; ];
            CellObj[cell]["next"] = Null;
        ]
    ]  
);


Options[CellObjEvaluate] = {
    "JSON" -> False,
    "callback" -> Null
};

CellObj /: 
CellObjEvaluate[CellObj[cell_], evaluator_, OptionsPattern[]] := (  
    Module[{exps = StringSplit[CellObj[cell]["data"], "\n"], parent},

        (*will break the chain if we try to evaluate a child cell*)
        If[(parent = CellObjFindParent[CellObj[cell]]) =!= Null,
            CellObjRemoveAllNext[CellObj[cell]];

            (*dont touch the previuos children*)
            If[CellObj[cell]["prev"] =!= Null, CellObj[cell]["prev"]["next"] = Null];

            (*reassign*)
            CellObj[cell]["prev"] = parent;
            CellObj[cell]["next"] = parent["next"];
            parent["next"]["prev"]  = CellObj[cell];
            parent["next"]          = CellObj[cell];
            
            CellObj[cell]["type"] = "input";
            CellObj[cell]["parent"] = Null;

            CellEventFire["CellMove"][CellObj[cell], parent];
            CellEventFire["CellMorph"][CellObj[cell]];
        ];    

        (*if has a child*)
        If[CellObj[cell]["child"] =!= Null,
            CellObjRemoveAllNext[CellObj[cell]["child"]];
            CellObjRemove[CellObj[cell]["child"]];
            CellObj[cell]["child"] = Null;            
        ];

        With[{callback = OptionValue["callback"]},
            evaluator[#, Function[result,
                With[{new = CellObjCreateChild[CellObj[cell] ]},
                    new["data"] = result;
                    new["type"] = "output";

                    CellEventFire["NewCell"][new];
                ]
            ] ]& /@ exps;
        ];
    ];  
);

