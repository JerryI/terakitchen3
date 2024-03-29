Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];

$host = CEval[$ev]["addr"];
$uid = CreateUUID[];

$data["list"] = CheckLocked[ $data["list"] ];
manualsamples = CQuery["$manual" :> Exist -> Equal -> True, $data["list"] ];

samples = CQuery["type" -> Equal -> "sample", $data["list"]];

If[Length[samples] == 0 && Length[manualsamples] == 0, PushNotification["services/ref","nothing to do..."]; $Die];

Module[{refs, sm}, 
  sm = CGet[#]; 
  console["log", "looking for "<>#<>"..."];

  query = {"type" -> Equal -> "ref"};

  Do[
    With[{field = key, val = sm[key]},
      AppendTo[query, {field -> Equal -> val, field :> Exist -> Equal -> False}:>Or];
    ]
          
  , {key, Complement[sm//Keys, {"type", "data", "date", "filename", "experiment", "id"}]}];

  refs = CQuery[query :> And, $data["list"]];
  

  sm["ref"] = If[Length[refs] == 0, "noref", console["log", "found "<>First[refs]<>""]; First[refs] ];
  

  With[{uid = sm["id"], ref = sm["ref"]},
    CEval[
      collection[uid, "ref"] = ref;
    ]
  ];
] & /@ samples;

(*fetching for the rest*)

norefs = CQuery[{"ref" -> Equal -> "noref"} :> And, samples]; 
norefs = Join[norefs, manualsamples];

If[Length[norefs] == 0, PushNotification["services/ref", StringTemplate["`` samples and references are connected"][samples//Length]]; $Die];

(*norefs*)

CreatePage[
    LoadPage["services/refs/views/missmatch.wsp"],
    "id"->$uid,
    "postHandler"->ProcessEventEmitt[$id]
];  

EventBind["submit", Function[{cli, data}, ConnectRest[data]; PageDelete[$uid]; $Die]];

PushNotification["services/ref", StringTemplate["We coundn't connect all samples and references. Please, try to do it manually for the remaining part <a class=\"btn btn-outline-info\" href=\"window.wsp?id=``\">Fix</a>"][$uid]];

ConnectRest[input_] := Module[{data = input},
  PushNotification["services/ref/debug", "action: "<>ToString[data, InputForm] ];
  If[data["action"] === "Ok",
    
    
    data["action"] = .;
    data["id"] = .;

    console["log", "applied manual"];

    With[{id = #, ref = data[#]},
      PushNotification["services/ref", "Connecting <br> <it>"<>ref<>"</it> to <it>"<>id<>"</it> as a reference"];
      CEval[collection[id, "ref"] = ref; collection[id, "type"] = "sample"; collection[id, "$manual"] = .; ];
    ]&/@Keys[data];

    itsref = CQuery[{"ref" -> Equal -> "itsref"} :> And, Join[samples, manualsamples] ]; 

    With[{id = #},
      PushNotification["services/ref", "Setting <it>"<>id<>"</it> as a reference"];
      CEval[collection[id, "type"] = "ref"; collection[id, "ref"] = .; collection[id, "$manual"] = .;];
    ]&/@itsref;    
  , 
    console["log", "discarded manual"];
    PushNotification["services/ref", "Cancelled"];

    With[{id = #},
      CEval[collection[id, "ref"] = .;];
    ]&/@norefs;
  ]
];
 


