input = $data["list"];

If[Length[input] == 0, PushNotification["Signal modeling", "nothing to do..."]; $Die];

Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];
Import["services/signalmodeling/functions.wls"];

samples = CQuery[{"type"->Equal->"sample"}:>And, input];
console["log", "number: "<>ToString[samples//Length]];

PhaseTrashhold = 5.3;

GetPhase[obj_] := Module[{t0, \[Phi]0, ph, right},
  t0 = (MaximalBy[obj["tds"]//Transpose, Abs[#[[2]]] &] // First // First) - obj["tds"][[1,1]];
  
  right = 
   Position[sample["freqs"], _?((Abs[# - 120] < 1) &)] // 
     First // First;

  \[Phi]0 = 2 \[Pi] (1/33.356) t0 # & /@ sample["freqs"];

  ph = (Arg[obj["fft"][[2]]] -  \[Phi]0);

  PadRight[Take[ClusterPhase[ph, 1, right, PhaseTrashhold], right], Length[ph] ]

];

fourier[data_] := Module[{tds, fft},
    tds = data;
    If[OddQ[data//Length], tds = Drop[data, -1]];
     
    fft = Fourier[tds]; 

    (*cut the unusable half*)
    Drop[fft, -Length[fft]/2]
];

ifourier[data_] := InverseFourier[Join[data, {0}, Drop[Reverse[Conjugate[data]], -1]]] // Re;

equalList[x_, y_] := If[Length[x] > Length[y], {x, Join[y, Table[0., {i, Length[x]-Length[y]}] ]}, If[Length[x] < Length[y], {x, Drop[y, Length[x] - Length[y] ]},{x,y}] ];

src["tds"] := (
    equalList[sample["tds"][[All,1]], ifourier[src["src"] ] ]
);

src["fft"] := (
    equalList[sample["freqs"], src["src"] ]
);

sam["compute"] := (

    sam["fft"] = {sample["freqs"], src["src"] Conjugate[GenTransmission]};
    sam["tds"] = equalList[sample["tds"][[All,1]], ifourier[sam["fft"][[2]] ] ];
);

GenTransmission := Switch[method, 
    "alpha",
    Module[{nk = sample["nk"], thickness, freqs},
        thickness = CGet[samid, "thickness"];
        freqs = sample["freqs"];

        If[MissingQ[ thickness ] || !StringQ[thickness], 
            thickness = With[{id = CGet[samid, "experiment"]}, CEval[ experiments[id, "thickness"] ] ];
        ];

        thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression[thickness] ];

        With[{transm = 
            If[!TrueQ[fpmodel], 
              Table[
                With[
                    {
                        L = thickness, \[Omega] = 2 \[Pi]  10^12 ((freqs[[i]]/33.356)) , c = 2.9979 10^10 , 
                        nair = 1,
                        n = nk[[i, 1]], k = nk[[i, 2]]
                    },

                     (4 (n - I k) nair)/((n - I k) + 
                        nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] (*1/(
                      1 - (((n - I k) - nair)/((n - I k) + 
                          nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])*)

                ]
              , {i, 1, Length[freqs]}]
            ,
              Table[
                With[
                    {
                        L = thickness, \[Omega] = 2 \[Pi]  10^12 ((freqs[[i]]/33.356)) , c = 2.9979 10^10 , 
                        nair = 1,
                        n = nk[[i, 1]], k = nk[[i, 2]]
                    },

                     (4 (n - I k) nair)/((n - I k) + 
                        nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] 1/(
                      1 - (((n - I k) - nair)/((n - I k) + 
                          nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])

                ]
              , {i, 1, Length[freqs]}]
            ]
        },    


            transm


        ]

    ],

    "transmission",
    (Exp[-I #[[2]]]Sqrt[#[[1]]])&/@sample["transmission"]
];



startDialog[uuid_, ssam_, ssamid_] := (
    $uid = uuid;
    sample = ssam;
    samid = ssamid;

    refid = CGet[samid, "ref"];

    reference = CGet[refid, "data"];

    $next = CreateUUID[];

    If[Length[samples] == 0, 
        postRedirect[next_] := Function[data, If[data["action"] === "Ok", "index.wsp", "index.wsp"] ];
    ,
        postRedirect[next_] := Function[data, If[data["action"] === "Ok", "window.wsp?id="<>next, "index.wsp"] ];
    ];

    method = "alpha";

    If[!KeyExistsQ[sample, "alpha"],
        PushNotification["Signalmodelling", "<span class=\"badge badge-info\">alpha field is absent</span> The transmittance filed will be used instead."];
        method = "transmission";
    ];   

    content =   Block[{}, 
                    LoadPage["services/signalmodeling/views/"<>method<>".wsp"]
                ];

    src["src"] = source[samid];

    WebSocketPublish[LoadDynamicPage[content], $uid];            

    CreatePage[
        content,
        "id"           -> $uid,
        "postRedirect" -> postRedirect[$next],
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    EventBind["submit", Function[{cli, data},

        With[{sid = samid, name = sourcename, rid = refid, samtds = sam["tds"][[2]], samfft = sam["fft"][[2]], reffft = src["fft"][[2]], reftds = src["tds"][[2]]},
            CEval[
                collection[sid, "data", "sigmodel-"<>name<>"-tds"] = samtds;
                collection[sid, "data", "sigmodel-"<>name<>"-fft"] = samfft;
            ];
            CEval[
                collection[sid, "data", "sigmodel-"<>name<>"-src-tds"] = reftds;
                collection[sid, "data", "sigmodel-"<>name<>"-src-fft"] = reffft;
            ];            
        ]; 

        If[Length[samples] == 0, $Die];
        startDialog[$next, CGet[samples//First, "data"], samples//First];
        samples = Drop[samples, 1];

        PushNotification["Signal modeling", "Dialog window &#9935; <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];
    ]];   

    EventBind["event", Function[{cli, data}, 
       

        Switch[data//First,
            "srcLoad",
                With[{sr = LoadPage["services/signalmodeling/views/sources/"<>data[[2]]<>".wsp"]},

                    WebSocketSend[cli, UpdateSource[sr] ]; 

                    src["src"] = source[samid];
                    
                    WebSocketSend[cli, UpdateSourceData[ ExportString[src["fft"]//Abs, "JSON"], ExportString[src["tds"], "JSON"], ExportString[src["phase"], "JSON"] ] ]; 
                ],

            "update",
                With[{},
                    data//Last//ReleaseHold;

                    src["src"] = source[samid];
                    WebSocketSend[cli, UpdateSourceData[ ExportString[src["fft"]//Abs, "JSON"], ExportString[src["tds"], "JSON"], ExportString[src["phase"], "JSON"] ] ];

                    sam["compute"];
                    WebSocketSend[cli, UpdateSampleData[ ExportString[sam["fft"]//Abs, "JSON"], ExportString[sam["tds"], "JSON"], ExportString[GetPhase[sam], "JSON"] ] ]; 
                
                ]
                
        
        ];
    ]];


);

fpmodel = True;

startDialog[CreateUUID[], CGet[samples//First, "data"], samples//First];
samples = Drop[samples, 1];

PushNotification["Signal modeling", "Dialog window &#9935; <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];