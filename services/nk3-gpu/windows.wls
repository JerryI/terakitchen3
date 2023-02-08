openDialogWindow["Check the parameters"][samid_] := Module[{pack,content, dataset, result=<||>},
    pack = GetTransmission[samid];
    pack["trust region"] = GetRegion[pack, {locut, hicut}];
    pack = UnwarpPhase[pack][pthresh];

    dataset = { {ThicknessNormalise["cm"][pack["thickness"] ], pack["scale"], 2Pi phcorr} };
    result["raw"]       = ProcessTransmission[pack, dataset, "drypasses"->0, "wetpasses"->0]["results"]//First;
    pack["data"]        = ProcessTransmission[pack, dataset, "drypasses"->Round[drycycles], "wetpasses"->Round[wetcycles] ];
    result["processed"] = pack["data"]["results"]//First;

    result["raw"]       = result["raw"][[pack["trust region"][[1]] ;; pack["trust region"][[2]]]];
    result["processed"] = result["processed"][[pack["trust region"][[1]] ;; pack["trust region"][[2]]]];

    pack["raw"]         = result["raw"];
    pack["dataset"]     = dataset;
    PackSupplimentary[samid]["number"] = 1;
    

    content = Block[{
                        scale = pack["scale"], 
                        thickness = ThicknessNormalise["mm"][pack["thickness"] ],
                        freqs = result["processed"][[All, 1]],
                        refractive = result["processed"][[All, 2]],
                        kappa  = result["processed"][[All, 3]],
                        rkappa = result["raw"][[All, 3]]
                    },

                LoadPage["services/nk3-gpu/views/index.wsp", {id = samid}] 
    ];

    Pack[samid] = pack;
    Clear[pack];

    (* danger *)
    MEMORYLimit = 10 200 1024 1024;

    (*clear eventlisterner*)
    EventBind["event", Function[{cli, data}, 
    Module[{vars, dataset, pack = Pack[samid]},
        Switch[data[[1]],

            "getThicknessPanel",
                WebSocketSend[cli, LoadPartPage[LoadPage["services/nk3-gpu/views/parts/thickness.wsp"] ] ]
                ,

            "update",
                vars = Association /@ ImportString[data[[2]], "JSON"];

                If[!ValidateVariables[vars], Return[$Failed, Null] ];
                ApplyVariables[vars];

                pack["trust region"]    = GetRegion[pack, {locut, hicut}];
                pack["thickness"]       = thickness/10.0;
                pack["scale"]           = scale;

                pack = UnwarpPhase[pack][pthresh];
                dataset = { {pack["thickness"], pack["scale"], 2Pi phcorr} };
                
                result["raw"]       = ProcessTransmission[pack, dataset, "drypasses"->-1, "wetpasses"->-1]["results"]//First;

                pack["data"]        = ProcessTransmission[pack, dataset, "drypasses"->Round[drycycles], "wetpasses"->Round[wetcycles] ];
                result["processed"] = pack["data"]["results"]//First;

                result["raw"]       = result["raw"][[pack["trust region"][[1]] ;; pack["trust region"][[2]]]];
                result["processed"] = result["processed"][[pack["trust region"][[1]] ;; pack["trust region"][[2]]]];
                WebSocketSend[cli, UpdateGraph[ result["processed"][[All, 1]],result["processed"][[All, 2]],result["processed"][[All, 3]],result["raw"][[All, 3]]  ] ];

                pack["raw"]         = result["raw"];
                pack["dataset"]     = dataset;

                Pack[samid] = pack;
                Clear[pack];
                ,

            "debug-reload",
                openDialogWindow["Check the parameters"][samid]
        ];
    ] 
    ] ];

    EventBind["submit", Function[{cli, data}, 
    Module[{types},
        If[data["action"] != "Ok", PageDelete[$uid]; $Die];
        PagePlaceholder[$uid];

        types = <|
                    "locut"     -> "number",
                    "hicut"     -> "number",
                    "thickness" -> "number",
                    "scale"     -> "number",
                    "pthresh"   -> "number",
                    "phcorr"    -> "number",
                    "drycycles" -> "number",
                    "wetcycles" -> "number",
                    "corrections" -> "bool",
                    "applyfp"     -> "bool",
                    "applyforall" -> "bool",

                    "varythicknessrange" -> "number",
                    "varyscalerange"     -> "number"
        |>;

        If[!ValidateVariables[data, types], Return[$Failed, Null] ];

        (*default*)

        corrections = False;
        applyfp = False;
        applyforall = False;

        ApplyVariables[data, types];


        If[!corrections, 
            UploadNK[samid];

            If[Length@samples === 0, PageDelete[$uid]; $Die];

            If[applyforall, 
                PackGlobal["set"] = Pack[samid]["dataset"][[ PackSupplimentary[samid]["number"] ]];
                PackGlobal["accuracy"] = Pack[samid]["accuracy"];

                PackGlobal["trust region"] = Pack[samid]["trust region"];

                ProcessNextSilently[samples//First];
                Return[Null, Module];
            ];

            
            Unset[PackSupplimentary[samid]["number"] ];
            Unset[Pack[samid] ];

            openDialogWindow["Check the parameters"][samples//First];
            samples = Drop[samples, 1];

            Return[Null, Module];
        ];
        

        openDialogWindow["Visor"][samid];
    ]
    ] ];    
    
    WebSocketPublish[LoadDynamicPage[content], $uid];
    CreatePage[
        content,
        "id"           -> $uid,
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    Warn["start"];
];


openDialogWindow["Visor"][samid_] := Module[{pack, content, dataset, data, opt, preliminary},
    pack = Pack[samid];
    pack["trust region"] = GetRegion[pack, {locut, hicut}];
    pack = UnwarpPhase[pack][pthresh];

    dataset = Table[{ {pack["thickness"](1.0 + t/100), pack["scale"](1.0 + s/100), 2Pi phcorr} }, {t, -varythicknessrange, varythicknessrange, varythicknessrange/25}, {s, -varyscalerange, varyscalerange, varyscalerange/10}];
    dataset = Partition[dataset//Flatten, 3];

    pack["dataset"] = dataset;
    pack["accuracy"] = pack["thickness"] varythicknessrange/25/100; 

    data = ProcessTransmission[pack, dataset, "drypasses"->drycycles, "wetpasses"->wetcycles];
    opt  = Position[data["2d-map"], MinimalBy[data["2d-map"], Last]//First]//First//First;
    If[!NumberQ[opt], opt = 0; PushNotification["services/nk3-gpu/windows", "<span class=\"badge badge-danger\">internal error</span> see <i>opt</i> variable"] ];

    preliminary = data["results"][[opt, pack["trust region"][[1]] ;; pack["trust region"][[2]]]];
    PackSupplimentary[samid]["graph"] = preliminary;
    PackSupplimentary[samid]["number"] = opt;
    
    content = Block[{
                        map     = data["2d-map"],
                        freqs  =  pack["raw"][[All,1]],
                        rkappa  = pack["raw"][[All,3]],

                        kappa   = preliminary[[All, 3]],
                        n       = preliminary[[All, 2]]
                    },

                LoadPage["services/nk3-gpu/views/visor.wsp", {id = samid}] 
    ];

    EventBind["event", Function[{cli, data}, 
    Module[{graph},
        Switch[data[[1]],
            "getGraph at",
                If[!NumberQ[data[[2]]], PushNotification["services/nk3-gpu/windows", StringTemplate["<span class=\"badge badge-danger\">internal error</span> `` is not a number"][data[[2]]] ]; Return[$Failed, Module]; ];
                PackSupplimentary[samid]["graph"] = Pack[samid]["data", "results"][[data[[2]], Pack[samid]["trust region"][[1]] ;; Pack[samid]["trust region"][[2]]]];
                PackSupplimentary[samid]["number"]= data[[2]];

                WebSocketSend[cli, ShowTHzGraph[PackSupplimentary[samid]["graph"][[All, 2]], PackSupplimentary[samid]["graph"][[All, 3]]] ];
                Clear[graph];
        ]
    ]
    ] ];

    WebSocketPublish[LoadDynamicPage[content], $uid];
    CreatePage[
        content,
        "id"           -> $uid,
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];   

    EventBind["submit", Function[{cli, data}, 
    Module[{types},
        If[data["action"] != "Ok", PageDelete[$uid]; $Die];
        PagePlaceholder[$uid];

        UploadNK[samid];

        If[Length@samples === 0, PageDelete[$uid]; $Die];

        If[applyforall, 
            PackGlobal["set"] = Pack[samid]["dataset"][[ PackSupplimentary[samid]["number"] ]];
            PackGlobal["accuracy"] = Pack[samid]["accuracy"];

            PackGlobal["trust region"] = Pack[samid]["trust region"];

            ProcessNextSilently[samples//First];
            Return[Null, Module];
        ];

        
        Unset[PackSupplimentary[samid]["number"] ];
        Unset[Pack[samid] ];

        openDialogWindow["Check the parameters"][samples//First];
        samples = Drop[samples, 1];
    ]
    ] ];

    pack["data"] = data; 
    Pack[samid] = pack;
    Clear[pack];
];

openDialogWindow["Phase"][samid_, callback_] := Module[{graphs},
    graphs = Pack[samid]["data", "results"][[All, Pack[samid]["trust region"][[1]] ;; Pack[samid]["trust region"][[2]], 2]];
    freqs  = Pack[samid]["data", "results"][[1, Pack[samid]["trust region"][[1]] ;; Pack[samid]["trust region"][[2]], 1]];

    If[(graphs//Length) =!= 3, PushNotification["services/nk3-gpu/windows", "<span class=\"badge badge-danger\">internal error</span> at <i>AskPhaseCorrection</i>. The number of graphs is not 3"]; Return[$Failed, Module] ];
    
    content = Block[{
                        graph1 = graphs[[1]],
                        graph2 = graphs[[2]],
                        graph3 = graphs[[3]],
                        xaxis  = freqs
                    },

                LoadPage["services/nk3-gpu/views/phase.wsp", {id = samid}] 
    ];

    WebSocketPublish[LoadDynamicPage[content], $uid];
    EventBind["event", Function[{cli, data}, Null ] ];

    CreatePage[
        content,
        "id"           -> $uid,
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    EventBind["submit", Function[{cli, data}, 
    Module[{types},
        If[data["action"] != "Continue", PushNotification["services/nk3-gpu", "Interrupted"];PageDelete[$uid]; $Die];
        PagePlaceholder[$uid];

        callback[data["option"]//ToExpression];
    ]
    ] ];    


];


Warn["start"] := (
    PushNotification["services/nk3-gpu", "Check the parameters <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];
    Warn["start"] = Null;
);