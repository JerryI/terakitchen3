(*cutoff = 0.2;
iterator = 1; 
globalpoints = {};

        wsp = Get["webserver/wsp.wls"];
        wsp["install"];

        $host = CEval[$host];

cutRegion[regions_, in_, round_ : 2] := 
    Module[{l, r, data = in, delta, reg},
        If[Length[regions] == 0, Return[in]];
        reg = regions // First;
        l = Position[data[[All, 1]], _?((Abs[# - reg[[1]]] < 1) &)] // First // First;
        r = Position[data[[All, 1]], _?((Abs[# - reg[[2]]] < 1) &)] // First // First;
        
        data[[r ;; r + round, 2]] = Join[data[[r ;; r, 2]], MovingAverage[data[[r ;; r + round, 2]], 2]];
        data[[l - round ;; l, 2]] = Join[MovingAverage[data[[l - round ;; l, 2]], 2], data[[l - round ;; l - round, 2]]];

        delta = (data[[r, 2]] - data[[l, 2]])/(r - l);

        Do[data[[i, 2]] = data[[l, 2]] + delta (i - l), {i, l + 1, r - 1}];

        cutRegion[Drop[regions, 1], data, round]
    ];

        SubBack[regions_, in_]:= With[{cutted =  cutRegion[regions, in] }, LowpassFilter[cutted[[All, 2]], cutoff]];

        PopupAskCheckRegion[sam_, ref_, thickness_, uuid_ : CreateUUID[], next_:"index.wsp"] := Module[{page=<||>, probe, points, resp, subglobalpoints=globalpoints}, 
           
            logWrite["ASK"];
            page["content"] = 
                Block[{id = uuid, cut = cutoff, alpha = Map[ {#[[1]], (((#[[1]] 4 \[Pi]  10^12 #[[3]])/(33.356 2.9979 10^10)))} &, Transpose[Join[{sam["freqs"]}, Transpose[sam["nk"]]]]], cutted = SubBack[globalpoints, Map[ {#[[1]], (((#[[1]] 4 \[Pi]  10^12 #[[3]])/(33.356 2.9979 10^10)))} &, Transpose[Join[{sam["freqs"]}, Transpose[sam["nk"]]]]]]}, 
                 StringJoin[
                    Map[
                        If[#[[1]] == "HTML", 
                            #[[2]], 
                            ToExpression[#[[2]]] /. {Null -> ""}
                        ] &, 
                        WSP[OpenRead["services/background/views/index.wsp"]]
                    ]
                  ]
                ];

            page["handler"] = Function[
                x,
                Module[{id = CreateUUID[], resp = x},
                    If[x["action"] == "Retry",
                        session["Redirect"] = "dynamic.wsp?id="<>id;
                        resp = Join[resp, <|"next"->id|>];
                    ];
                    If[x["action"] == "Cancel",
                        session["Redirect"] = "index.wsp";
                    ];                        
                        
                    resp
                ]
            ];

            page["redirect"] = next;  

            With[{page = page}, CEval[ 
                    
                window[uuid] = page; 
                window[uuid]["query"] = CreateDataStructure["Queue"];

                WebSocketPublish[LoadDynamicPage[window[uuid]["content"]], uuid];
                    
            ]];

            PushNotification["Background", "Entscheidung erforderlich. Select the range to cut the modes <br>"<>"<a class=\"btn btn-outline-info\" href=\"http://"<>$host<>"/dynamic.wsp?id="<>uuid<>"\">open</a>"];
            
            While[CEval[!KeyExistsQ[window[uuid],"result"]],

                If[CEval[!window[uuid]["query"]["EmptyQ"]],
                    probe = CEval[window[uuid]["query"]["Pop"]];
                    
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    logWrite["PPPPPPPPPPPPPUUSHH"];
                    Module[{pts, result},
                        cutoff = probe[[1]];

                        subglobalpoints = With[{new = ImportString[probe[[2]],"JSON"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];
                        
                        With[{dt = SubBack[subglobalpoints, Map[ {#[[1]], (((#[[1]] 4 \[Pi]  10^12 #[[3]])/(33.356 2.9979 10^10)))} &, Transpose[Join[{sam["freqs"]}, Transpose[sam["nk"]]]]] ]},
                            CEval[ WebSocketSend[window[uuid]["socket"], UpdateGraph[ExportString[dt,"JSON"]] ];  ];
                        ];
                        
                    ];
                ];        
                Pause[0.2];
            ];
            
            points = CEval[ImportString[window[uuid]["result"]["points"],"JSON"]];
            
            cutoff = CEval[window[uuid]["result"]["cut"]//ToExpression];
            
            resp = CEval[window[uuid]["result"]];

            CEval[window[uuid] = .;];
           
           
            
            <|"points"-> points, "responce"->resp|>
        ];

        samples = CQuery[{"type"->Equal->"sample"}:>And, input];
        logWrite["number: "<>ToString[samples//Length]];   

            (*ask about the range*)
        nxt = CreateUUID[];

        iterator = 1;
       
        While[iterator <= Length[samples],

            With[{sam = CGet[#], ref = With[{uid = #}, CEval[collection[collection[uid]["ref"]]]]},
                Module[{reply},
                    
                    reply = PopupAskCheckRegion[sam["data"], ref["data"], Null, nxt, If[iterator == Length[samples], "index.wsp", "dynamic.wsp?id="<>(nxt =  CreateUUID[]) ]]; 

                    logWrite[reply["responce"]]
                    Switch[reply["responce"]["action"],
                        "Ok",
                            logWrite["ok good"];
                            iterator = iterator + 1;
                            globalpoints = With[{new = reply["points"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];
                        
                            With[{r = SubBack[globalpoints, Map[ {#[[1]], (((#[[1]] 4 \[Pi]  10^12 #[[3]])/(33.356 2.9979 10^10)))} &, Transpose[Join[{sam["data"]["freqs"]}, Transpose[sam["data"]["nk"]]]]]], uid = #},
                                CEval[
                                    collection[uid]["data"]["background"] = r;
                                    collection[uid]["data"]["subtracted"] := With[{id = uid}, Transpose[{collection[id]["data"]["freqs"],collection[id]["data"]["alpha"][[All,2]] - collection[id]["data"]["background"]}]];
                                ];
                            ]
                        ,
                        "Retry",
                            logWrite["try again"];
                            nxt = reply["responce"]["next"];
                        ,
                        "Cancel",
                            iterator = Length[samples] + 1;
                            logWrite["cennceld"];
                    ]
                    logWrite["DONE"];
                ]

            ]&@samples[[iterator]];
        
        ];        


    
    ]

]*)
PushNotification["Background", "indev.. skipped"];

$Die;