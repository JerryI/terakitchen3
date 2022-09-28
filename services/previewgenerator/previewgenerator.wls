experiments = <||>;
(experiments[CGet[#, "experiment"]] = True) &/@ $data["list"];

(*debug*)
$Messages = {OpenWrite[Method -> "Toast"]};

With[{exp = #, graphs = ToExpression@graphs, steps = ToExpression@steps, width = ToExpression@width},
    With[{lst = CQuery[{"type" -> Equal -> "sample", "experiment" -> Equal -> exp} :> And, $data["list"]]},

        CEval[
            experiments[exp]["preview"] = ExportString[
            
            ListLinePlot[

                With[{c = collection[#]["data"]["alpha"][[;; ;; steps]]}, If[ArrayQ[c], c, {{0,0}} ] ] &/@ If[Length[lst] < graphs, lst, lst[[1 ;; graphs]] ]
            
            , Axes -> False, ImageSize -> width, AspectRatio->0.7], "SVG"]   

        ];

    ]
]&/@(Keys@experiments);

PushNotification["Previews", "Generated"];

$Die;