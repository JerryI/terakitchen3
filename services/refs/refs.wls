Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];

$host = CEval[$ev]["addr"];
$uid = CreateUUID[];
  
samples = CQuery["type" -> Equal -> "sample", $data["list"]];

If[Length[samples] == 0, PushNotification["Ref","nothing to do..."]; $Die];

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
  console["log", "found "<>First[refs]<>""];

  sm["ref"] = If[Length[refs] == 0, "noref", First[refs]];

  CPut[sm];
] & /@ samples;

(*fetching for the rest*)

norefs = CQuery[{"ref" -> Equal -> "noref"} :> And, samples]; 

If[Length[norefs] == 0, PushNotification["Ref", StringTemplate["`` samples and references are connected"][samples//Length]]; $Die];

(*norefs*)

CreatePage[
    LoadPage["services/refs/views/missmatch.wsp"],
    "id"->$uid,
    "postHandler"->ProcessEventEmitt[$id]
];  

EventBind["submit", Function[{cli, data}, ConnectRest[data]; $Die]];

PushNotification["Ref", StringTemplate["We coundn't connect all samples and references. Please, try to do it manually for the remaining part <a class=\"btn btn-outline-info\" href=\"window.wsp?id=``\">Fix</a>"][$uid]];

ConnectRest[input_] := Module[{data = input},
  If[data["action"] === "Ok",
    
    data["action"] = .;
    data["id"] = .;

    console["log", "applied manual"];

    With[{id = #, ref = data[#]},
      CEval[collection[id, "ref"] = ref;];
    ]&/@Keys[data];

    norefs = CQuery[{"ref" -> Equal -> "noref"} :> And, samples]; 

    With[{id = #},
      CEval[collection[id, "type"] = "ref";];
    ]&/@norefs;    
  , 
    console["log", "discarded manual"];

    With[{id = #},
      CEval[collection[id, "ref"] = .;];
    ]&/@norefs;
  ]
];
 


