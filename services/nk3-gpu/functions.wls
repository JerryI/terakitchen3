GetRegion[data_, trustreg_] := Module[{left, right},
    left = Position[data["transmission"][[All, 1]], _?((Abs[# - trustreg[[1]]] < 0.5) &)] // First // First;
    right = Position[data["transmission"][[All, 1]], _?((Abs[# - trustreg[[2]]] < 0.5) &)] // First // First;

    If[!NumberQ[left],  PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">Error</span> `` is beyong the range"][trustreg[[1]] ] ]; left = 70; ];
    If[!NumberQ[right], PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">Error</span> `` is beyong the range"][trustreg[[2]] ] ]; right = 512-70; ];

    {left, right}
];

GetTransmission[samid_String] := Module[{sam, ref, thickness, transmission, nInitial},   
    sam = CGet[samid, "data"];
    ref = CGet[CGet[samid, "ref"], "data"];

    If[!StringQ[ CGet[samid, "ref"] ] || With[{uid = CGet[samid, "ref"]}, CEval[!KeyExistsQ[collection, uid] ] ] ,
        PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">error</span> the reference signal is absent. Try to use service <a href=\"/service.wsp?id=refs&q='id'->Equal->'``'\" class=\"badge badge-info\">refs</a> first"][samid] ];
        PageDelete[$uid];
        $Die;        
    ];

    CheckFields[samid, {"tds", "freqs", "fft"}];
    CheckFields[CGet[samid, "ref"], {"tds", "freqs", "fft"}];

    (With[{r = Evaluate[sam[#] ]}, sam[#] = r]) &/@ {"tds", "freqs", "fft"}; 
    (With[{r = Evaluate[ref[#] ]}, ref[#] = r]) &/@ {"tds", "freqs", "fft"};
    thickness = CEval[ GetThickness["cm"][samid] ];

    If[!NumberQ[thickness], 
        PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">fatal error</span> `` is not a proper thickness value. Change it and try again"][thickness] ];
        PageDelete[$uid];

        (*one could ask it from the push up and correct it*)
        $Die;
    ];

    sam["t0"] = (MaximalBy[sam["tds"], Abs[#[[2]]] &] // First // First) - sam["tds"][[1, 1]];
    ref["t0"] = (MaximalBy[ref["tds"], Abs[#[[2]]] &] // First // First) - ref["tds"][[1, 1]];
    sam["ph0"] = 2 \[Pi] (1/33.356) sam["t0"] # & /@ sam["freqs"];
    ref["ph0"] = 2 \[Pi] (1/33.356) ref["t0"] # & /@ ref["freqs"];
    nInitial = 1 + (0.029979*(-1.*ref["t0"] + sam["t0"])/thickness);
    transmission = {0, Abs[#], 0, 0} & /@ (sam["fft"]/ref["fft"]);
    transmission[[All, 1]] = sam["freqs"];
    transmission = Take[transmission, 1024]; 

    <|
        "original length"->Length[sam["freqs"] ],
        "transmission"->transmission, 
        "n0"->nInitial, "thickness"->thickness, "scale"->1.0,
        "raw phase difference"->Take[(Arg[sam["fft"]] - sam["ph0"]) - (Arg[ref["fft"]] - ref["ph0"]), 1024],
        "constant phase difference" -> Take[sam["ph0"] - ref["ph0"], 1024],
        "trust region" -> {70, 512 - 70}
    |>
];

UnwarpPhase[data_][PhaseThreshold_] := Module[{transmission, out = data},
    transmission = data["transmission"];
    transmission[[All, 3]] = ClusterPhase[out["raw phase difference"], 1, 1024-1, PhaseThreshold] + out["constant phase difference"];
    out["transmission"] = transmission;
    out
];

Options[ProcessTransmission] = {"drypasses"->0, "wetpasses"->8, "skips"->2};

ProcessTransmission[data_, dataset_, OptionsPattern[]] := Module[{dest, src, set, results, trunc, power, skips = OptionValue["skips"]},
    dest = CUDAMemoryLoad[Table[0, {i,1, Length[dataset](8*1024 + 8)}], "Float"];
    src  = CUDAMemoryLoad[Join[data["transmission"]//Flatten, {data["n0"], data["thickness"], 1.0, 0Pi}], "Float[4]"];
    set = CUDAMemoryLoad[dataset,  "Float[3]"];

    results = Partition[ Partition[CUDAMemoryGet[autorun[src, dest, OptionValue["drypasses"], OptionValue["wetpasses"], set, Length[dataset]*1024]//First],8], 1025];

    CUDAMemoryUnload/@ {dest, src, set};
    
    trunc = Function[x,(#[[{1,3}]] &/@ x[[ data["trust region"][[1]] ;; data["trust region"][[2]] ]] )]/@ results;
    power = (Fourier[#[[All,2]]]//Abs//DropHalf)&/@trunc;
    dt = (Total[Drop[#, skips ] ])&/@ power;
    
    <| "2d-map" -> (Join[Take[Transpose[dataset],2], {dt}]), "results" -> results |>
];


UploadNK[samid_] := Module[{data, raw, set},
    data = Join[Pack[samid]["data"]["results"][[PackSupplimentary[samid]["number"], ;;-2, {2,3}  ]], Table[{Pack[samid]["n0"], 0.0}, {i, Pack[samid]["original length"]-1024}] ];
    set  = Pack[samid]["dataset"][[ PackSupplimentary[samid]["number"] ]];
    raw  = Join[ProcessTransmission[Pack[samid], { set }, "drypasses"->-1, "wetpasses"->-1]["results"][[1, ;;-2, {2,3} ]], Table[{Pack[samid]["n0"], 0.0}, {i, Pack[samid]["original length"]-1024}] ];

    With[{uid = samid, rawk = raw[[All,2]], accuracy = Pack[samid]["accuracy"], range = Sequence[Pack[samid]["trust region"][[1]] ;; Pack[samid]["trust region"][[2]] ], thickness = set[[1]], scale = set[[2]], nk = data},
      CEval[
        collection[uid, "attributes", "usable-range"] = range;
        collection[uid, "attributes", "thickness-accuracy"] = accuracy;
        collection[uid, "attributes", "virtual-aperture"] = scale;
        collection[uid, "thickness"]                    = ToString[10.0 thickness]<>" mm";
        collection[collection[uid, "ref"], "thickness"] = ToString[10.0 thickness]<>" mm";

        collection[uid, "data"]["nk"] = nk;

        collection[uid, "data"]["alpha-with-fp"] = With[{}, 
                                        ((#[[2]] 4 \[Pi]  10^12 #[[1]])/(33.356 2.9979 10^10)) & /@ 
                                            Transpose[{collection[uid]["data"]["freqs"], rawk}]
                                        ];

        collection[uid]["data"]["alpha"] := With[{id = uid}, 
                                        {#[[1]], (#[[2]] 4 \[Pi]  10^12 #[[1]])/(33.356 2.9979 10^10)} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["nk"][[All, 2]]}]
                                        ];
        collection[uid]["data"]["n"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["nk"][[All, 1]]}]
                                        ];
        collection[uid]["data"]["k"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["nk"][[All, 2]]}]
                                        ];
      ];        
    ];
];

ValidateVariables[vars_] := Module[{},
    Do[
        If[!TrueQ[(<|"bool"->BooleanQ, "number"->NumberQ, "text"->StringQ|>[i["type"] ])[(<|"bool"->ToExpression, "number"->ToExpression, "text"->Identity|>[i["type"] ])[i["value"] ] ] ],
            PushNotification["services/nk3-gpu/windows", StringTemplate["<span class=\"badge badge-danger\">error</span> `` is not a `` type"][i["value"], i["type"] ] ];
            Return[False, Module];
        ]
    , {i, vars}];

    Return[True, Module];
];

ValidateVariables[assoc_, types_] := Module[{},
  If[Not[TrueQ[<|"text"->StringQ, "number"->Function[x, ToExpression[x]//NumberQ], "bool"->Function[x, If[x === "on", True, False] ] |>[If[KeyExistsQ[types, #], types[#], "text"] ][assoc[#] ] ] ],
   PushNotification["services/nk3-gpu/windows", StringTemplate["<span class=\"badge badge-danger\">error</span> `` is not a `` type"][assoc[#], types[#] ] ];
   Return[False, Module];
  ]&/@Keys[assoc];

  Return[True, Module];
];

ApplyVariables[vars_] := Do[StringTemplate[<|"bool"->"`` = ``;", "number"->"`` = ``;", "text"->"`` = \"``\";"|>[i["type"] ] ][i["id"], i["value"] ]//ToExpression, {i, vars}];

ApplyVariables[vars_, types_] := ( StringTemplate[<|"bool"->"`` = If[\"``\" === \"on\", True, False];", "number"->"`` = ``;", "text"->"`` = \"``\";"|>[ If[KeyExistsQ[types, #], types[#], "text"] ] ][ #, vars[#] ]//ToExpression )&/@Keys[vars];

CheckFields[uid_String, fields_List] := Module[{failed = <||>},
    With[{id = uid, field = #}, If[!TrueQ[CEval[ KeyExistsQ[collection[id, "data"], field] ] ],  failed[#]=True; ] ] &/@ fields;
    Switch[#,
        "fft",
            PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">error</span> field <i>fft</i> is absent. Use service <a href=\"/service.wsp?id=fourier&q='id'->Equal->'``'\" class=\"badge badge-info\">fourier</a> first"][uid] ];        
        ,
        "freqs",
            PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">error</span> field <i>fft</i> is absent. Use service <a href=\"/service.wsp?id=fourier&q='id'->Equal->'``'\" class=\"badge badge-info\">fourier</a> first"][uid] ];
        ,
        "tds",
            PushNotification["services/nk3-gpu/functions", StringTemplate["<span class=\"badge badge-danger\">error</span> field <i>tds</i> is absent, which is usually not a case. Try to delete and upload your sample data again. ``"][""] ];
    ]&/@Keys[failed];
    If[Length[Keys[failed] ] =!= 0, PageDelete[$uid]; $Die; ];
];

