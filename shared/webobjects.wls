SVGForm[x_] := ExportString[x, "SVG"]//HTMLForm;

JSRun[x_String, name_String:"JS"] := (StringTemplate["<div class=\"badge badge-danger\">``</div><script>``</script>"][name, x])//HTMLForm;

Options[ListPloty] = {
    PlotRange -> {{-Infinity, Infinity}, {-Infinity, Infinity}}
};

ListPloty[x_List, OptionsPattern[]] := Module[{transp},
    Switch[Depth[x],
        3,
            transp = { (Transpose[FilterRange[x, OptionValue[PlotRange] ] ])}
        ,
        4,
            transp = ( Transpose[FilterRange[#, OptionValue[PlotRange] ] ] ) &/@ x
        ,
        2,
            transp = {Table[i, {i, 1, Length[x]}], x}
    ];

    WListPloty[ExportString[transp//N, "JSON"] ]
];

Options[Ploty] = {
    PlotRange -> {{-Infinity, Infinity}, {-Infinity, Infinity}}
};

Ploty[function_, range_, OptionsPattern[] ] := Module[{transp, data},
    With[{ite = range[[1]], it = Join[range, {range[[3]]-range[[2]]}/200//N]},
        With[{fn = #},
            Table[{ite, fn}, it ]
        ] &/@ Flatten[{function}]
    ]//ListPloty
];

WebExport[name_, exp_, frmt_:Automatic] := (
    Export["public/trashcan/"<>name, exp, frmt];
    StringTemplate["<a href=\"trashcan/``\" class=\"badge badge-warning\">Download ``</a>"][name, name]//HTMLForm
);