input = $data["list"];
extrapolatePhase = "no";
phaseCorr = 0;
amplify = 1;
thicknessAccuracy = 0; 

If[Length[input] == 0, PushNotification["Transmittance","nothing to do..."]; $Die];



Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];
Import["services/transmittance/functions.wls"];

samples = CQuery[{"type"->Equal->"sample"}:>And, input];
console["log", "number: "<>ToString[samples//Length]];

(*global thickness*)
thickness = With[{id = First[input]}, CEval[ Block[{cm = 1, mm = 0.1, um = 10^(-4)}, experiments[collection[id]["experiment"], "thickness"]//ToExpression ]  ] ];
console["log", "T calculations: "<>ToString[thickness]<>"  cm - thickness"];



(*take a random sample*)
Do[
    With[{sam = CGet[i], ref = With[{uid = i}, CEval[collection[collection[uid]["ref"]]]]},
        If[KeyExistsQ[sam,"thickness"],
            console["log", "custom thickness: "<>sam["thickness"]];
            thickness = Block[{mm = 0.1, cm = 1, um = 10^(-4)}, ToExpression[sam["thickness"]] ];
        ];

        If[With[{uid = i}, CEval[!KeyExistsQ[collection, collection[uid]["ref"]]]], 
            PushNotification["T extractor", StringTemplate["Problem with `` has occured. No reference was found."][i]]; 
            Continue[];
        ];

        {probesam, proberef} = {sam, ref};
        samid = i;

        Break[];
    ]
, {i, samples}];

$uid = CreateUUID[];

startDialog[uuid_] := (
    $uid = uuid;

    result = ExtractTransmittance[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}];


    
    $next = CreateUUID[];

    If[askall,
        If[Length[samples] == 0,
            postRedirect[next_] := Function[data, If[data["action"] == "Ok", "index.wsp", "index.wsp"] ];
        ,
            postRedirect[next_] := Function[data, If[data["action"] == "Ok", "window.wsp?id="<>next, "index.wsp"] ];
        ];
    ,
        postRedirect[next_] := Function[data, If[data["action"] == "Ok", "index.wsp", "index.wsp"] ];
    ];
    

    content =   Block[{amp=amplify, ep=extrapolatePhase, ph=phaseCorr, lo=locutoff, hi=hicutoff, transmission=result["transmission"], fq = probesam["data"]["freqs"]}, 
                    LoadPage["services/transmittance/views/index.wsp"]
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

        {locutoff, hicutoff, phaseCorr, amplify, PhaseTrashhold} = ToExpression/@data/@{"lo","hi","ph","amp", "phtresh"};

        (*check for errors*)
        (If[!NumberQ[#//N], Throw[StringTemplate["`` is not a number at checking the range"][#]]]) &/@ {locutoff, hicutoff, phaseCorr, amplify};

        If[askall,
            
            result = ExtractTransmittance[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}];
            store[samid];


            If[Length[samples] == 0, $Die];

            With[{i = First@samples},
                With[{sam = CGet[i], ref = With[{uid = i}, CEval[collection[collection[uid]["ref"] ] ] ]},    
                    {probesam, proberef} = {sam, ref};
                    samid = i;      
                ]        
            ];

            samples = Drop[samples, 1];
            
            startDialog[$next];


            
                      
        ,
            ApplyToAll;
            $Die;        
        ];

    ]];

    EventBind["event", Function[{cli, data}, 
        {locutoff, hicutoff, phaseCorr, amplify, PhaseTrashhold} = data;

        result = ExtractTransmittance[probesam["data"], proberef["data"], thickness, {locutoff, hicutoff}];
      
        With[{dt = {probesam["data"]["freqs"], result["transmission"][[All,1]], result["transmission"][[All,2]]}//N },

            CEval[ddbug = dt;];

            WebSocketSend[cli, UpdateGraph[ExportString[dt,"JSON"] ] ];

        ]; 
    ]];

    PushNotification["Transmittance", "Entscheidung erforderlich. Check the range <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];
);


store[i_] := (
                With[{uid = i, r = result["transmission"], index = result["refractive index"]},
                    CEval[ 

                        collection[uid]["data"]["transmission"] = r; 

                        collection[uid]["refractive-index"] = index; 

                        collection[uid]["data"]["transmittance"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ];
                        collection[uid]["data"]["absorbance"] := With[{id = uid}, 
                                        {#[[1]], 2-Log10@#[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ];                                        
                        collection[uid]["data"]["phase-shift"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 2]]}]
                                        ];
                        collection[uid]["data"]["rough-alpha"] := With[{id = uid}, 
                                        With[{d = GetThickness["cm"][id]},
                                            {#[[1]], -Log[#[[2]]]/d} & /@ 
                                                Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ] ];                                               

                        ];
                ];

           
                With[{uid = i},
                    CEval[
        
                        1+1 "LOL"
                    ]
                ];   

);


ApplyToAll := (
    Do[
        With[{sam = CGet[i], ref = With[{uid = i}, CEval[collection[collection[uid]["ref"]]]]},

            If[With[{uid = i}, CEval[!KeyExistsQ[collection, collection[uid]["ref"]]]], 
                PushNotification["Transmittance", StringTemplate["Problem with `` has occured. No reference was found."][i]]; 
                Continue[];
            ];

            If[KeyExistsQ[sam,"thickness"],
                    console["log", "custom thickness: "<>sam["thickness"]];
                    thickness = Block[{mm = 0.1, cm = 1, um = 10^(-4)}, ToExpression[sam["thickness"]] ];
            ];

            If[amplify != 1,
                With[{uid = i, amp = amplify//N//DecimalForm}, 
                    CEval[ 
                        collection[uid, "tds-scale"] = amp;
                        collection[collection[uid, "ref"], "tds-scale"] = amp;
                    ]; 
                ];
            ];
            

            Module[{virgin},

                result = ExtractTransmittance[sam["data"], ref["data"], thickness,  {locutoff, hicutoff}];


                With[{uid = i, r = result["transmission"], index = result["refractive index"]},
                    CEval[ 

                        collection[uid]["data"]["transmission"] = r; 

                        collection[uid]["refractive-index"] = DecimalForm[index // N, 3]; 

                        collection[uid]["data"]["transmittance"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ];
                        collection[uid]["data"]["phase-shift"] := With[{id = uid}, 
                                        {#[[1]], #[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 2]]}]
                                        ];
                        collection[uid]["data"]["absorbance"] := With[{id = uid}, 
                                        {#[[1]], 2-Log10@#[[2]]} & /@ 
                                            Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ];  

                        collection[uid]["data"]["rough-alpha"] := With[{id = uid}, 
                                        With[{d = GetThickness["cm"][id]},
                                            {#[[1]], -Log[#[[2]]]/d} & /@ 
                                                Transpose[{collection[id]["data"]["freqs"], collection[id]["data"]["transmission"][[All, 1]]}]
                                        ] ];                                                                               

                        ];
                ];

           
                With[{uid = i},
                    CEval[
        
                        collection[uid]["data"]["tds-no-fp"] = .;
                    ]
                ];                   
         

            ]
        ];

    , {i, samples}];   

    PushNotification["Transmittance", "Done! <style>
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
If[askall, samples = Drop[samples, 1] ];

