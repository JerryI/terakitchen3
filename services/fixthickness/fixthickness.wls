If[Not[ThicknessNormalise[units]["1 mm"]//NumberQ//TrueQ],
  PushNotification["services/fixthickness", "entered value "<>ToString[ThicknessNormalise[units]["1 mm"] ]<>" is <span class=\"badge badge-warning\">not a correct unit</span> We will use mm instead."]
  units = "mm";
];

(
  With[{id = #, u = units},
    CEval[
      If[KeyExistsQ[collection[id], "thickness"],
        collection[id, "thickness"] = StringTemplate["`` ``"][ThicknessNormalise[u][collection[id, "thickness"] ], u];
      ];
    ]
  ]
) &/@ $data["list"];

$Die;