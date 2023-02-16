cutoff = 0.2;
iterator = 1; 
globalpoints = {};

$data["list"] = CheckLocked[ $data["list"] ];
If[Length[$data["list"]] == 0, PushNotification["Filter","nothing to do..."]; $Die];


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

        Do[data[[i, 2]] = 0.0, {i, l + 1, r - 1}];

        cutRegion[Drop[regions, 1], data, round]
    ];

SubBack[regions_, in_]:= With[{cutted =  cutRegion[regions, in] }, cutted[[All, 2]] ];

GenTDS[transm_] := Module[{}, 
    With[{sim = ref["fft"] transm}, InverseFourier[Join[sim, {0}, Drop[Reverse[Conjugate[sim] ], -1] ] ] // Re]
];

startDialog[uuid_, ssam_, ssamid_] := (
    $uid = uuid;
    sam = ssam;
    samid = ssamid;

  
    ref = CGet[samid, "ref"];
    If[!StringQ[ref], PushNotification["Filter", "No refs found! 🛠"]; $Die;];
    ref = CGet[ref, "data"];

    If[!AssociationQ[ref],  PushNotification["Filter", "No refs found! 🛠"]; $Die;];


    $next = CreateUUID[];

    If[Length[samples] > 1,
        postRedirect[next_] := Function[d, If[d["action"] === "Ok", "window.wsp?id="<>next, "index.wsp"]]; 
    ,
        postRedirect[next_] := Function[d, If[d["action"] === "Ok", "index.wsp", "index.wsp"]]; 
    ];

    

    transm = sam["fft"]/ref["fft"];

    freqs  = sam["freqs"];
    cutted = SubBack[globalpoints, Transpose[{freqs, transm}]];
    

    content =   Block[{}, 
                    LoadPage["services/filter/views/index.wsp"]
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

        PushNotification["Filter", "Not implemented"];
        $Die;

        globalpoints = With[{new = ImportString[data["points"],"JSON"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];
        cutoff = ToExpression[data["cut"]];

        cutted = SubBack[globalpoints, Transpose[{freqs, k}]];

        With[{uid = samid, r = cutted},
            CEval[ 
                (*collection[uid, "data", "Filter"] = r;    *)  
                        
            
            
            ];
                           
        ];

        

        If[Length[samples] == 0,
            PushNotification["Filter", "Done"];
            $Die;
        ];

        startDialog[$next, CGet[samples//First, "data"], samples//First];
        samples = Drop[samples, 1];


    ]];   

    EventBind["event", Function[{cli, data}, 
        cutoff = ToExpression[data[[1]]];
        globalpoints = With[{new = ImportString[data[[2]],"JSON"]},  If[Length[new] == 0, globalpoints,  (MinMax /@ FindClusters[new, CriterionFunction -> "CalinskiHarabasz"])]];

        cutted = SubBack[globalpoints, Transpose[{freqs, transm}]];     

        WebSocketSend[cli, UpdateGraph[ExportString[cutted//Abs, "JSON"] ] ];

        WebSocketSend[cli, UpdateTDS[ExportString[GenTDS[cutted], "JSON"]] ];
    ]];


);

startDialog[CreateUUID[], CGet[samples//First, "data"], samples//First];
samples = Drop[samples, 1];

PushNotification["Filter", "Open an editor <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];


            