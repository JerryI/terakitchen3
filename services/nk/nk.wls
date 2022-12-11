input = $data["list"];
extrapolatePhase = "no";
phaseCorr = 0;
amplify = 1;
thicknessAccuracy = 0; 

If[Length[input] == 0, PushNotification["NK","nothing to do..."]; $Die];

(*debug*)
If[debug, $Messages = {OpenWrite[Method -> "Toast"]} ];

Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];
Import["services/nk/functions.wls"];

samples = CQuery[{"type"->Equal->"sample"}:>And, input];
console["log", "number: "<>ToString[samples//Length]];

(*global thickness*)
thickness = With[{id = First[input]}, CEval[ Block[{cm = 1, mm = 0.1, um = 10^(-4)}, experiments[collection[id]["experiment"], "thickness"]//ToExpression ]  ] ];
console["log", "NK calculations: "<>ToString[thickness]<>"  cm - thickness"];

(*take a random sample*)
Do[
    With[{sam = CGet[i], ref = With[{uid = i}, CEval[collection[collection[uid]["ref"]]]]},
        If[KeyExistsQ[sam,"thickness"],
            console["log", "custom thickness: "<>sam["thickness"]];
            thickness = Block[{mm = 0.1, cm = 1, um = 10^(-4)}, ToExpression[sam["thickness"]] ];
        ];

        If[With[{uid = i}, CEval[!KeyExistsQ[collection, collection[uid]["ref"]]]], 
            PushNotification["NK extractor", StringTemplate["Problem with `` has occured. No reference was found."][i]]; 
            Continue[];
        ];

        {probesam, proberef} = {sam, ref};

        probesam["data", "freqs"]   = probesam["data", "freqs"];
        probesam["data", "fft"]     = probesam["data", "fft"];
        probesam["data", "tds"]     = probesam["data", "tds"];
        proberef["data", "freqs"]   = proberef["data", "freqs"];
        proberef["data", "fft"]     = proberef["data", "fft"];
        proberef["data", "tds"]     = proberef["data", "tds"];     
        If[KeyExistsQ[probesam["data"], "unwrapped-phase"], probesam["data", "unwrapped-phase"] = With[{uuid = probesam["id"] }, CEval[ collection[uuid]["data"]["unwrapped-phase"] ] ];  ];   
        
        

        samid = i;

        Break[];
    ]
, {i, samples}];

$uid = CreateUUID[];

startDialog[uuid_] := (
    $uid = uuid;

    result = ExtractNK[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}];

    If[Min[result[[2]][[All,2,1]]] < 0, Warn["anomaly refractive index"] ];
    
    nofp = If[numberFP > 0, First[result[[1]]]["first"], {}];
    $next = CreateUUID[];

    console["log", "thicness correction: ``", thicknessCorrection];



    If[thicknessCorrection, 
        postRedirect[next_] := Function[data, If[data["action"] == "Ok", "window.wsp?id="<>next, "index.wsp"]],
        postRedirect[next_] := Function[data, If[data["action"] == "Ok", "index.wsp", "index.wsp"]]
    ];

    content =   Block[{amp=amplify, ep=extrapolatePhase, ph=phaseCorr, fp=numberFP, fpnumber=numberFP, lo=locutoff, hi=hicutoff, alpha = Transpose[Map[({ #[[1]], ((#[[2,2]] 4 \[Pi]  10^12 #[[1]])/(33.356 2.9979 10^10))})&, result[[2]]]], alphanofp = Transpose[Map[({ #[[1]], ((#[[2,2]] 4 \[Pi]  10^12 #[[1]])/(33.356 2.9979 10^10))})&, nofp]], n=result[[2]][[All,2,1]]}, 
                    LoadPage["services/nk/views/askrange.wsp"]
                ];

    WebSocketPublish[LoadDynamicPage[content], $uid];            

    CreatePage[
        content,
        "id"           -> $uid,
        "postRedirect" -> postRedirect[$next],
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    EventBind["submit", Function[{cli, data}, 
        If[data["action"] != "Ok", $Die];

        {locutoff, hicutoff, numberFP, phaseCorr, amplify, PhaseTrashhold} = ToExpression/@data/@{"lo","hi","fp","ph","amp", "phtresh"};

        (*check for errors*)
        (If[!NumberQ[#//N], Throw[StringTemplate["`` is not a number at checking the range"][#]]]) &/@ {locutoff, hicutoff, numberFP, phaseCorr, amplify};


        If[!thicknessCorrection,
            ApplyToAll;

            $Die;
        ,
            $uid = $next;
            $next = CreateUUID[];

            result = ExtractNK[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}][[2]];

            content = Block[{lo=locutoff, hi=hicutoff, alpha = Transpose[Map[({ #[[1]], ((#[[2,2]] 4 \[Pi]  10^12 #[[1]])/(33.356 2.9979 10^10))})&, result]]}, 
                LoadPage["services/nk/views/askfp.wsp"]
            ];

      

            WebSocketPublish[LoadDynamicPage[content], $uid];

            CreatePage[
                content,

                "id"           -> $uid,
                "postRedirect" -> postRedirect[$next],
                "postHandler"  -> ProcessEventEmitt[$id]
            ];

            PushNotification["NK", "Entscheidung erforderlich. Select the range with the less absorbtion, where the FP oscillations are clear <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];



            EventBind["submit", Function[{cli, data},  
                deviation = data["dev"]//ToExpression;
                recursionNumber = data["recur"]//ToExpression; 

                (*checkregion = (MinMax /@ FindClusters[points, CriterionFunction -> "CalinskiHarabasz"]);*)
                checkregion = (MinMax /@ FindClusters[ImportString[data["points"],"JSON"], CriterionFunction -> "CalinskiHarabasz"]);

                (*check for errors*)
                (If[!NumberQ[#//N], Throw[StringTemplate["`` is not a number at FP stage"][#]]]) &/@ {deviation, recursionNumber};


                startVisor[$next, thickness (1 - deviation/100), thickness (1 + deviation/100), thickness (2 deviation/100 )/15];
            ]];

        ];
    ]];

    EventBind["event", Function[{cli, data}, 
        {locutoff, hicutoff, numberFP, phaseCorr, amplify, PhaseTrashhold} = data;

        result = ExtractNK[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}];

        If[Min[result[[2]][[All,2,1]]] < 0, Warn["anomaly refractive index"] ];

        nofp = If[numberFP > 0, First[result[[1]]]["first"], {{0,{0,0}}}];
        With[{dt = DropHalf/@{result[[2]][[All,1]],((#[[2, 2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10)) &/@result[[2]], result[[2]][[All,2,1]],  ((#[[2, 2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10)) &/@nofp}},

            WebSocketSend[cli, UpdateGraph[ExportString[dt,"JSON"]] ];

        ]; 
    ]];

    PushNotification["NK", "Entscheidung erforderlich. Check the range <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];
);

Warn["anomaly refractive index"] := (
    PushNotification["NK", "<span class=\"badge badge-warning\">Anomaly refractive index</span> was detected which might create an artificial features in the absorption spectrum. Please, consider to cut the noisy part of the spectrum or use <a class=\"badge badge-info\" href=\"service.wsp?id=transmittance\">transmittance</a> service instead."];
    Warn["anomaly refractive index"] = Null;
);

Warn["phase was unwrapped"] := (
    PushNotification["NK", "<span class=\"badge badge-info\">The phase was alredy unwrapped</span> We will skip this procedure for the sample signal."];
    Warn["phase was unwrapped"] = Null;
);


startVisor[uuuid_, ll_, rr_, st_] := (
    $uid = uuuid;
    $next = CreateUUID[];

    postRedirect[next_] := Function[data, If[data["action"]==="Ok" || data["action"]==="Cancel" || data["action"]==="Continue without FP subtraction", "index.wsp", "window.wsp?id="<>next]];
    content = LoadPage["services/nk/views/visor.wsp"];
    (*clear eventlisterner*)
    EventBind["event", Function[{cli, data}, Null;]];

    WebSocketPublish[LoadDynamicPage[content], $uid];
    CreatePage[
        content,
        "id"           -> $uid,
        "postRedirect" -> postRedirect[$next],
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    PushNotification["NK", "Webvisor. Live preview by the link <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];
    
    EventBind["submit", Function[{cli, data},
        Switch[data["action"],
            "Cancel",
                TaskAbort[process];
                $Die,

            "Ok",
                thickness = L;
                ApplyToAll; 
                $Die,

            "Retry",
                TaskAbort[process];
                startDialog[$next];,

            "More iterations",
                recursionNumber = 1;
                PushNotification["NK", "One more round"];
                startVisor[$next, lL, rL, step];,

            "Continue without FP subtraction",
                numberFP = 0;
                ApplyToAll; 
                $Die,

            "Continue with a new thickness",
                thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression[data["thickness"]]];
                amplify = ToExpression[data["amp"]];
                
                (*check for errors*)
                (If[!NumberQ[#//N], Throw[StringTemplate["`` is not a number. thickness"][#]]]) &/@ {thickness, amplify};

                variants = {};

                startVisor[$next, thickness (1 - deviation/100), thickness (1 + deviation/100), thickness (2 deviation/100 )/15];,

            "Optimize amplitude",
                thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression[data["thickness"]]];
                amplify = ToExpression[data["amp"]];
                
                (*check for errors*)
                (If[!NumberQ[#//N], Throw[StringTemplate["`` is not a number. thickness"][#]]]) &/@ {thickness, amplify};

                variants = {};

                startVisor[$next, thickness, thickness, thickness];
        ];
    ]];

    (*create temporary stack*)
    stack = {};

    (*client asked for the previous data*)
    EventBind["event", Function[{cli, data},
        WebSocketSend[cli, UpdateGraphMulti[stack//Flatten]];
    ]];

    variants = {};
    process = LoopSubmit[ correctThickness[ll, rr, st] ];
);


correctThickness[left_, right_, istep_] := (
    lL = left; 
    rL = right;
    step = istep;
    L = thickness;

    Switch[
    Module[{},
      If[lL == rL,

        If[This["status"] != "running", Return["skip", Module]];

        Do[
            If[This["status"] != "running", Break[]];

            amplify = scale;

            ProcessNK@First@ExtractNK[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}, checkregion];

        , {scale, amplify 0.5, 1.5 amplify, 0.05 amplify}];
        
        variants = variants // Flatten;

        sorted = SortBy[Select[({#["amplitude"], #["iteration"], #["error"]}) & /@ Flatten[variants], #[[2]] == (numberFP) &], First];
        errorWithFP = (First@MinimalBy[Select[({#["amplitude"], #["iteration"], #["error"]}) & /@ Flatten[variants], #[[2]] == 0 &], Last])[[3]];
        minimal = First @ First @ Position[#, MinimalBy[#, Last] // First] &@ sorted;
        amplify = sorted[[minimal, 1]];  

        Return["done", Module];



      
      ,
        (*go recursively*)
        Do[
            trials = 0;
            (*probe every point*)

            Do[
                If[This["status"] != "running", Break[]];

                (*skip existing*)
                (*If[(Length@Select[#["thickness"] & /@ Flatten[variants], (Abs[# - LL] < (LL/15/8)) &]) > 0, Continue[]];*)

                ProcessNK@First@ExtractNK[probesam["data"], proberef["data"], LL, {locutoff, hicutoff}, checkregion];
            , {LL, lL, rL, step}];

            While[trials < 3,
                If[This["status"] != "running", Break[]];
                variants = variants // Flatten;

                (*analyse it*)
                sorted = SortBy[Select[({#["thickness"], #["iteration"], #["error"]}) & /@ Flatten[variants], #[[2]] == (numberFP) &], First];
                errorWithFP = (First@MinimalBy[Select[({#["thickness"], #["iteration"], #["error"]}) & /@ Flatten[variants], #[[2]] == 0 &], Last])[[3]];
                minimal = First @ First @ Position[#, MinimalBy[#, Last] // First] &@ sorted;
                L = sorted[[minimal, 1]]; 

                With[{dL = step},
                    If[(minimal <= 2 || Length[sorted] - minimal < 2) && k > 0,
                        
                        trials ++ ;
                        If[minimal <= 2,
                            Do[
                                lL = lL - dL;
                                If[This["status"] != "running", Return["skip", Module]];
                                ProcessNK@First@ExtractNK[probesam["data"], proberef["data"], lL, {locutoff, hicutoff}, checkregion];
                            , {j, 1,2}];
                        ,
                            Do[
                                rL = rL + dL;
                                If[This["status"] != "running", Return["skip", Module]];
                                ProcessNK@First@ExtractNK[probesam["data"], proberef["data"], rL, {locutoff, hicutoff}, checkregion];
                            , {j, 1,2}];
                        ];

                        PushNotification["NK", StringTemplate["Search region was extended `` +/- `` mm"][10(lL + rL)/2//N, 10(rL - lL)/2//N]];
                    ,
                        Break[];
                    ];
                ];
            ];

            If[This["status"] != "running", Return["skip", Module]];

            (*check number of trials*)
            If[trial >= 3,
                Return["nominima", Module];
            ];

            (*check if it actually works*)
            If[errorWithFP - sorted[[minimal, 3]] < 0,
                Return["phaseerror", Module];
            ];

            lL = (lL + L)/2;
            rL = (rL + L)/2;

            step = (rL - lL)/15;


        , {k, 1, recursionNumber}]; 

        thicknessAccuracy = ToString[10 step/2 //N// DecimalForm]<>" mm";
        thickness = L;
        Return["done", Module];
      ];

    ]   
    ,
    
    "nominima",
        EventBind["event", Function[{cli, data}, WebSocketSend[cli, UpdateGraphMulti[stack//Flatten]]; WebSocketSend[cli, NoMinima[]];]];
        WebSocketPublish[NoMinima[], $uid]; 
        PushNotification["NK", "No minimum was found. The procedure was suspended.<br><a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];,
    
    "phaseerror",
        EventBind["event", Function[{cli, data}, WebSocketSend[cli, UpdateGraphMulti[stack//Flatten]]; WebSocketSend[cli, PhaseError[errorWithFP - sorted[[minimal, 3]]]];]];
        WebSocketPublish[PhaseError[errorWithFP - sorted[[minimal, 3]]], $uid];
        PushNotification["NK", "Overcompensation was detected. FP subtraction failed. <br><a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];,
    
    "done",
        EventBind["event", Function[{cli, data}, Null;]];
        WebSocketPublish[ItIsOK[ToString[10 L //N//DecimalForm]<>" mm", amplify], $uid];

        With[{min = MinimalBy[ 
                    Select[({#["thickness"], #["iteration"], #["first"], #["error"]}) & /@ 
                            Flatten[variants], #[[2]] == (numberFP) &], Last] // First,
              virgin = MinimalBy[ 
                    Select[({#["thickness"], #["iteration"], #["first"], #["error"]}) & /@ 
                            Flatten[variants], #[[2]] == 0 &], Last] // First              
                            },
            With[{r = {min[[3]][[All,1]], ((#[[2, 2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10)) &/@(min[[3]]), min[[3]][[All,2,1]], ((#[[2, 2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10)) &/@(virgin[[3]])}},
                stack = ExportString[r, "JSON"];
                WebSocketPublish[UpdateFinalGraph[stack], $uid]; 
            ];
            (*client asked for the previous data*)
            EventBind["event", Function[{cli, data},
                WebSocketSend[cli, ItIsOK[ToString[10 L //N//DecimalForm]<>" mm", amplify] ];
                WebSocketSend[cli, UpdateFinalGraph[stack] ];
            ]];
        ];  

        PushNotification["NK", "Optimal thickness is "<>ToString[10 L //N//DecimalForm]<>" mm. TDS scaling is "<>ToString[amplify//N//DecimalForm]<>"<br><a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];,
    
    "skip",
        PushNotification["NK", "Stopped"];   
    ];

);

ApplyToAll := (
    Do[
        Module[{sam = CGet[i], ref = With[{uid = i}, CEval[collection[collection[uid]["ref"]]]]},

            If[With[{uid = i}, CEval[!KeyExistsQ[collection, collection[uid]["ref"]]]], 
                PushNotification["NK extractor", StringTemplate["Problem with `` has occured. No reference was found."][i]]; 
                Continue[];
            ];

            sam["data", "freqs"]   = sam["data", "freqs"];
            sam["data", "fft"]     = sam["data", "fft"];
            sam["data", "tds"]     = sam["data", "tds"];
            ref["data", "freqs"]   = ref["data", "freqs"];
            ref["data", "fft"]     = ref["data", "fft"];
            ref["data", "tds"]     = ref["data", "tds"];

            If[KeyExistsQ[sam["data"], "unwrapped-phase"], sam["data", "unwrapped-phase"] = With[{uuid = sam["id"] }, CEval[collection[uuid]["data"]["unwrapped-phase"] ] ]; ];             

            If[!thicknessCorrection || numberFP == 0,
                If[KeyExistsQ[sam,"thickness"],
                    console["log", "custom thickness: "<>sam["thickness"]];
                    thickness = Block[{mm = 0.1, cm = 1, um = 10^(-4)}, ToExpression[sam["thickness"]] ];
                ];
            ,
                With[{uid = i, t = thickness, thicknessAccuracy0 = thicknessAccuracy}, 
                    CEval[ 
                        collection[uid, "thickness"] = StringTemplate["`` mm"][DecimalForm[10 t//N]];
                        collection[collection[uid, "ref"], "thickness"] = StringTemplate["`` mm"][DecimalForm[10 t//N]];
                        collection[uid, "thickness-error"] = thicknessAccuracy0;
                        collection[collection[uid, "ref"], "thickness-error"] = thicknessAccuracy0;
                    ]; 
                ];
            ];

            If[amplify != 1,
                With[{uid = i, amp = amplify//N//DecimalForm}, 
                    CEval[ 
                        collection[uid, "tds-scale"] = amp;
                        collection[collection[uid, "ref"], "tds-scale"] = amp;
                    ]; 
                ];
            ];
            

            Module[{result,virgin},

                result = ExtractNK[sam["data"], ref["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}];

                (*
                
                With[{
                    
                    res = Table[
                        phaseCorr = ph;
                        result = ExtractNK[sam["data"], ref["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}];

                        With[{nkt = result[[2,All,2]]},
                            Take[nkt[[All,1]], 100]
                        ]

                    , {phaseCorr, {4Pi, 2Pi, 0, -2Pi, -4Pi}}]},
                    
                    ExportString[ListLinePlot[res, Ticks -> None, Axes -> None, PlotRange -> Full, PlotLegends -> {4Pi, 2Pi, 0, -2Pi, -4Pi}, ImageSize -> Tiny], "SVG"]
                ]
                
                
                
                *)


                With[{uid = i, r = result[[2,All,2]]},
                    CEval[ 

                        collection[uid]["data"]["nk"] = r; 

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

                If[numberFP > 0, 
                    virgin = 
                        Select[({#["thickness"], #["iteration"], #["first"], #["error"]}) & /@ 
                                result[[1]], (#[[2]] == 0 && #[[1]] == thickness) &] // First;
                    
                    With[{uid = i, ta = thicknessAccuracy, al = ((#[[2, 2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10)) &/@(virgin[[3]]), tds = TDSwithoutFP[sam["data"], ref["data"], thickness, {locutoff, hicutoff}, {locutoff, hicutoff}, result[[2]]]},
                        CEval[
                            collection[uid]["data"]["alpha-with-fp"] = al;
                            collection[uid]["data"]["tds-no-fp"] = tds;
                        ]
                    ];
                ,
                    With[{uid = i},
                        CEval[
                            collection[uid]["data"]["alpha-with-fp"] = .;
                            collection[uid]["data"]["tds-no-fp"] = .;
                        ]
                    ];                   
                ];

            ]
        ];

    , {i, samples}];   

    PushNotification["NK", "Done! <style>
                                    .wave {
                                      animation-name: wave-animation;  /* Refers to the name of your @keyframes element below */
                                      animation-duration: 2.5s;        /* Change to speed up or slow down */
                                      animation-iteration-count: infinite;  /* Never stop waving :) */
                                      transform-origin: 70% 70%;       /* Pivot around the bottom-left palm */
                                      display: inline-block;
                                    }

                                    @keyframes wave-animation {
                                        0% { transform: rotate( 0.0deg) }
                                       10% { transform: rotate(14.0deg) }  /* The following five values can be played with to make the waving more or less extreme */
                                       20% { transform: rotate(-8.0deg) }
                                       30% { transform: rotate(14.0deg) }
                                       40% { transform: rotate(-4.0deg) }
                                       50% { transform: rotate(10.0deg) }
                                       60% { transform: rotate( 0.0deg) }  /* Reset for the last half to pause */
                                      100% { transform: rotate( 0.0deg) }
                                    }
                                </style>
                                <span class=\"wave\">&#x1F44B;</span>
    "];

);


startDialog[$uid];

