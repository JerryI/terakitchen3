cutoff = 0.2;
iterator = 1; 
globalpoints = {};

$data["list"] = CheckLocked[ $data["list"] ];
If[Length[$data["list"]] == 0, PushNotification["services/background","nothing to do..."]; $Die];

$uid = CreateUUID[];
Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];

samples = CQuery[{"type"->Equal->"sample"}:>And, $data["list"]];
console["log", "number: "<>ToString[samples//Length]];

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


GenTDS[cutted_] := Module[{nk = sam["nk"], thickness},
    thickness = CGet[samid, "thickness"];

    If[MissingQ[ thickness ] || !StringQ[thickness], 
        thickness = With[{id = CGet[samid, "experiment"]}, CEval[ experiments[id, "thickness"] ] ];
    ];

    thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression[thickness] ];

    With[{transm = Table[
        With[
            {
                L = thickness, \[Omega] = 2 \[Pi]  10^12 ((freqs[[i]]/33.356)) , c = 2.9979 10^10 , 
                nair = 1,
                n = nk[[i, 1]], k = nk[[i, 2]] - If[freqs[[i]] > 0.0001, (0.0795758 cutted[[i]]/freqs[[i]]), 0.]
            },
     
             (4 (n - I k) nair)/((n - I k) + 
                nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] (*1/(
              1 - (((n - I k) - nair)/((n - I k) + 
                  nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])*)

        ], {i, 1, Length[freqs]}]
    },    


        With[{sim = ref["fft"] Conjugate[transm]}, InverseFourier[Join[sim, {0}, Drop[Reverse[Conjugate[sim] ], -1] ] ] // Re]

    
    ]

];

startDialog[uuid_, ssam_, ssamid_] := (

    sam = ssam;
    samid = ssamid;

    If[tdswrite,
    
        ref = CGet[samid, "ref"];
        If[!StringQ[ref], PushNotification["services/background", "No refs found! 🛠"]; tdswrite = False; ];
        ref = CGet[ref, "data"];

        If[!AssociationQ[ref],  PushNotification["services/background", "No refs found! 🛠"]; tdswrite = False; ];
    ];

    

    If[Length[samples] > 1,
        postRedirect[next_] := Function[d, If[d["action"] === "Ok", "window.wsp?id="<>next, "index.wsp"]]; 
    ,
        postRedirect[next_] := Function[d, If[d["action"] === "Ok", "index.wsp", "index.wsp"]]; 
    ];

    alpha  = With[{id = samid}, CEval[ collection[id, "data", "alpha"][[All,2]] ]];
    freqs  = sam["freqs"];
    cutted = SubBack[globalpoints, Transpose[{freqs, alpha}]];
    

    content =   Block[{}, 
                    LoadPage["services/background/views/index.wsp"]
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
        If[data["action"] != "Ok", PageDelete[$uid]; $Die];
        PagePlaceholder[$uid];

        globalpoints = With[{new = ImportString[data["points"],"JSON"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];
        cutoff = ToExpression[data["cut"]];

        cutted = SubBack[globalpoints, Transpose[{freqs, alpha}]];

        With[{uid = samid, r = cutted},
            CEval[ 
                collection[uid, "data", "background"] = r;  

                collection[uid]["data"]["subtracted"] := With[{id = uid}, 
                                            With[{alpha = collection[id, "data", "alpha"]},
                                                Transpose[{alpha[[All, 1]], alpha[[All, 2]] - collection[id, "data", "background"]}]
                                            ] 
                                        ];        
                        
            
            
            ];
                           
        ];

        

        If[Length[samples] == 0,
            PushNotification["services/background", "Done"];
            PageDelete[$uid];
            $Die;
        ];

        startDialog[$next, CGet[samples//First, "data"], samples//First];
        samples = Drop[samples, 1];


    ]];   

    EventBind["event", Function[{cli, data}, 
        cutoff = ToExpression[data[[1]]];
        globalpoints = With[{new = ImportString[data[[2]],"JSON"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];

        cutted = SubBack[globalpoints, Transpose[{freqs, alpha}]];     

        WebSocketSend[cli, UpdateGraph[ExportString[cutted, "JSON"]] ];

        If[tdswrite,
            WebSocketSend[cli, UpdateTDS[ExportString[GenTDS[cutted], "JSON"]] ];
        ];
    ]];


);

startDialog[CreateUUID[], CGet[samples//First, "data"], samples//First];
samples = Drop[samples, 1];

PushNotification["services/background", "Open an editor <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];


            