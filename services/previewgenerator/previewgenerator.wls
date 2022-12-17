experiments = <||>;
(experiments[CGet[#, "experiment"]] = True) &/@ $data["list"];

(*debug*)
$Messages = {OpenWrite[Method -> "Toast"]};

(*With[{exp = #, graphs = ToExpression@graphs, steps = ToExpression@steps, width = ToExpression@width},
    With[{field = With[{uid = #}, If[CEval[KeyExistsQ[collection[uid, "data"], "alpha"] ], "alpha", "rough-alpha"] ], lst = CQuery[{"type" -> Equal -> "sample", "experiment" -> Equal -> exp} :> And, $data["list"]]},
        
        With[{uid = #}, 
            If[!CEval[KeyExistsQ[collection[uid, "data"], field]//TrueQ ], 
                PushNotification["Preview generator", StringTemplate["<span style=\"badge badge-warning\">There is a problem</span> with <i>alpha</i> field of <a style=\"badge badge-info\" href=\"item.wsp?id=``\">``</a>. The preview will not be generated."][#] ];
                $Die;
            ]; 
        ]; 

        CEval[
            experiments[exp]["preview"] = ExportString[
            
            ListLinePlot[

                With[{c = collection[#]["data"][field][[;; ;; steps]]}, If[ArrayQ[c], c, {{0,0}} ] ] &/@ If[Length[lst] < graphs, lst, lst[[1 ;; graphs]] ]
            
            , Axes -> False, ImageSize -> width, AspectRatio->0.7], "SVG"]   

        ];

    ]
]&/@(Keys@experiments);*)

PushNotification["Preview generator", "Sorry. This service is under the maintenence."];

$Die;