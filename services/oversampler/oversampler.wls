$data["list"] = CheckLocked[ $data["list"] ];
input = $data["list"];

If[Length[input] == 0, PushNotification["services/oversampler","nothing to do..."]; $Die];

resample[data_, "Average", "Linear"] := Module[{nPoints, ifun, delta, t0, t1},
    {t0, t1} = #[[1]] & /@ data[[{1, -1}]];
    delta = t1 - t0;
    nPoints = Length@data;
    ifun = Interpolation[data, InterpolationOrder->1];

    Table[{t0 + i/nPoints*delta, ifun[t0 + i/nPoints*delta]}, {i, 0, nPoints - 1}]
];




(*backing up the original TD trace*)
With[{uid = #}, CEval[ 
    If[KeyExistsQ[collection[uid, "data"], "origtds"], collection[uid, "data", "tds"] = collection[uid, "data", "origtds"], collection[uid, "data", "origtds"] = collection[uid, "data", "tds"] ]; 
  ] ]&/@ input;


supress = False; 

(*check consistensy*)
With[{tds = CGet[#, "data", "tds"]},
  If[(std = StandardDeviation@Differences@(tds[[All,1]])) > 0.00008,

    
    With[{plot = If[supress, "", ExportString[ListLinePlot[Differences[tds[[All,1]]], AxesLabel->"ps", ImageSize->Small, AspectRatio->1.2],"SVG"]<>"<br> <br>" ] },
      PushNotification["services/oversampler", StringTemplate["<span class=\"badge badge-warning\">Time jitter</span> was detected in <a href=\"item/index.wsp?id=``\" class=\"badge badge-light\">``</a>.  `` The standart deviation is <span class=\"badge badge-info\">``E5</span>. ``"][#,#,plot, Round[std,0.0000001]100000, If[irregularfix//TrueQ, "A linear approximation will be applied to the average.", ""] ] ];
    ];

    supress = True;


    If[irregularfix//TrueQ,
      With[{new = resample[tds, "Average", "Linear"], id = #}, 
        CEval[collection[id, "data", "tds"] = new; ]
      ];
    ];
  ]; 
]& /@ input;

windows = With[{uid = #}, CEval[ collection[uid, "data", "tds"][[{1,-1}, 1]]//Differences//Abs//Last ] ]&/@ input;  
initial = With[{uid = #}, CEval[ collection[uid, "data", "tds"][[ 1, 1 ]] ] ]&/@ input;  
lengths = With[{uid = #}, CEval[ collection[uid, "data", "tds"]//Length ] ]&/@ input; 
steps   = With[{uid = #}, CEval[ (collection[uid, "data", "tds"][[{1,-1}, 1]]//Differences//Abs//Last)/(Length[collection[uid, "data", "tds"]] - 1) ] ]&/@ input; 

PushNotification["services/oversampler", 
"<table>
    <thead>
        <tr>
            <th>id</th>
            <th>step, ps</th>
            <th>window, ps</th>
        </tr>
    </thead>
    <tbody>"<>StringJoin[Table[StringTemplate["
        <tr>
            <td><a href=\"``\">``</a></td>
            <td><span class=\"badge badge-info\">``</span></td>
            <td><span class=\"badge badge-warning\">``</span></td>
        </tr>"][input[[n]], input[[n]], steps[[n]], windows[[n]]], {n, 1, Length[windows]}] ]<>"
    </tbody>
</table>"];

(*ifuns   = Interpolation/@Table[CGet[x, "data", "tds"], {x, input}];*)
ifuns   = With[{uid = #}, Interpolation[ CEval[ collection[uid, "data", "tds"] ], Method->method, InterpolationOrder->Round[order] ]  ]&/@input;

newstep   = If[oversampling//TrueQ, 0.5, 1] Min[steps];
newwindow = Max[windows];

With[{if = ifuns},
  CEval[ ifuns = if; ];
];


PushNotification["services/oversampler", StringTemplate["Resampling all to <span class=\"badge badge-info\">`` ps</span> with <span class=\"badge badge-info\">`` ps</span> window"][newstep, newwindow] ];



Do[ 
  With[{
      tdsu = Table[
        {
          i + initial[[n]],

          If[i > windows[[n]], 0.0, ifuns[[n]][ i + initial[[n]] ]  ]
        },

        {i, 0, newwindow, newstep}],

      uid = input[[n]]
    },

    CEval[ collection[uid, "data", "tds"] = tdsu; collection[uid, "attributes", "usable-range"] = Sequence[All]; ]

  ]
, {n, 1, Length[input]}];

(*checking the lengths*)
lengths = With[{uid = #}, CEval[ collection[uid, "data", "tds"]//Length ] ]&/@ input; 

PushNotification["services/oversampler", "The new lengths are "<>StringJoin[Table[StringTemplate["<span class=\"adge badge-light\">``</span> "][i], {i, lengths}] ]<>" points. Done"];

(*check consistensy*)
With[{tds = CGet[#, "data", "tds"]},
  If[(std = StandardDeviation@Differences@(tds[[All,1]])) > 0.00008,

    
    With[{plot = If[supress, "", ExportString[ListLinePlot[Differences[tds[[All,1]]], AxesLabel->"ps", ImageSize->Small, AspectRatio->1.2],"SVG"]<>"<br> <br>" ] },
      PushNotification["services/oversampler", StringTemplate["<span class=\"badge badge-warning\">Irregular samplerate</span> was detected in <a href=\"item/index.wsp?id=``\" class=\"badge badge-light\">``</a>.  `` The standart deviation is <span class=\"badge badge-info\">``E5</span>. A linear approximation will be applied to the average."][#,#,plot, Round[std,0.0000001]100000] ];
    ];

    supress = True;


    
    With[{new = resample[tds, "Average", "Linear"], id = #}, 
      CEval[collection[id, "data", "tds"] = new; ]
    ];
  ]; 
]& /@ input;

$Die;


