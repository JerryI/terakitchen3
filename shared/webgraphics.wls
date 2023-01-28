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
            transp = x;
    ];

    WListPloty[ExportString[transp//N, "JSON"] ]
];

Options[Ploty] = {
    PlotRange -> {{-Infinity, Infinity}, {-Infinity, Infinity}}
};

Ploty[function_, range_, OptionsPattern[] ] := Module[{transp, data},
    With[{ite = range[[1]], it = Join[range, {range[[3]]-range[[2]]}/200//N]},
    
        data = Table[{ite, function}, it ];
    ];
    transp = { (Transpose[FilterRange[data, OptionValue[PlotRange] ] ])}

    WListPloty[ExportString[transp//N, "JSON"] ]
];