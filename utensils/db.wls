(*loading DB*)

dbbases = (First@
     StringCases[#, 
      RegularExpression["((.*-)(\\d*))"] :> <|"filename" -> "$1", 
        "ver" -> ToExpression["$3"]|>] & /@ 
   Select[FileNames["*", "db"], 
    Length[StringCases[#, "default-"]] > 0 &]);

If[Length@dbbases == 0, console["log", "no db files was found. Abort"]; Exit[]];

dbbases = SortBy[dbbases, -#["ver"] &];

While[TrueQ[FileExistsQ[(First@dbbases)["filename"] <> ".lock"]],
  DeleteFile[(First@dbbases)["filename"] <> ".lock"];
  console["log", "db `` crashed. skipping...", (First@dbbases)["filename"] ];
  dbbases = Drop[dbbases, 1];
];

dbbases = First@dbbases;
console["log", "openning ``", dbbases ];

Get/@FileNames["*.mx", dbbases["filename"]];

(*check the secret string*)
If[secretstring =!= "TERAPUSS",
  console["log", "secret string did not match! DB might be currupted. "];
  Exit[-1];
];

(*storage*)

SGlobalDate = Now;


Set[collection[x_, "data", y_], r_] ^:= storage["set", x, y, r];
Set[collection[x_, "data"][y_], r_] ^:= storage["set", x, y, r];
Set[collection[x_] ["data"][y_], r_] ^:= storage["set", x, y, r];
Unset[collection[x_] ["data"], y_] ^:= storage["remove", x, y];
Unset[collection[x_, "data"], y_] ^:= storage["remove", x, y];
Unset[collection[x_, "data", y_] ] ^:= storage["remove", x, y];
Unset[collection[x_]["data"][y_] ] ^:= storage["remove", x, y];
Unset[collection[x_, "data"][y_] ] ^:= storage["remove", x, y];


storage["url"]="archive/";

storage["set",id_,field_, data_]:=With[{},
  console["log", "storage >> setting `` for ``", field, id];
  If[!KeyExistsQ[collection[id, "data"], field],
    console["log", "storage >> `` doesnt exist. Creating...", field];
    With[{uid = CreateUUID[]},
      storage[uid]["data"]=data;
      storage[uid]["id"]=uid;
      storage[uid]["date"]=SGlobalDate;
      storage[uid]["archived"]=False;

      collection[id]["data"][field]:=storage["get", uid];
    ]
  ,

    With[{s=Extract[collection[id, "data"],{Key[field],2}]},
      console["log", "storage >> `` is rewritting... `` ", s, field];
      storage[s]["date"]=SGlobalDate;
      storage[s]["archived"]=False;
      storage[s]["data"]=data;
    ]
  ];
  data

];

storage["remove",id_,field_]:=With[{},
  console["log", "storage >> removing `` for ``", field, id];
  If[ KeyExistsQ[collection[id, "data"], field],
    With[ {obj= storage[Extract[collection[id,"data"],{Key[field],2}] ]},

      If[obj["archived"],
        DeleteFile[storage["url"]<>obj["id"]<>".mx"]
      ];

      Unset[obj["data"] ];
      Unset[obj["id"] ];
      Unset[obj["date"] ];
      Unset[obj["archived"] ];
    ];
    console["log", "storage >> `` for `` was removed", field, id];
    collection[id,"data"]=KeyDrop[collection[id,"data"],field ];
  ];
  Null
];

storage["get", uid_]:=(storage[uid]["date"]=SGlobalDate; storage[uid]["data"]);

CArchive := Module[{cnt = 0, now = Now}, (If[! #["archived"],
    If[now - #["date"] > Quantity[5, "Days"],
     console["log", "archived ``", #["id"]];
     cnt = cnt + 1;
     Print["archived " <> #["id"]];
     With[{url = 
        Export[storage["url"] <> #["id"] <> ".mx", #["data"]], str = "restored "<>#["id"]}, 
        #["data"] := (
        Print[str];
        #["archived"] = False;
        #["date"] = now;
        #["data"] = Import[url]
        )];
     #["archived"] = True;
     ]]; #["id"]) & /@ 
 DeleteDuplicates@(Extract[#, {1}, Extract[#, {1, 0}] &] & /@ 
    SubValues[storage]);

    PushNotification["Archiver", StringTemplate["`` items were unloaded to HDD"][cnt] ];
];

CArchiveStat := Module[{cnt = 0, toarchived = 0, archived = 0, now = Now}, 
(
  If[#["archived"],
    archived = archived + 1;
  ,
    If[now - #["date"] > Quantity[2, "Days"],
      toarchived = toarchived + 1;
    ,
      cnt = cnt + 1;
    ]
  ]
) & /@ 
 DeleteDuplicates@(Extract[#, {1}, Extract[#, {1, 0}] &] & /@ 
    SubValues[storage]);

    {cnt, toarchived, archived}
];

(*db operations*)
InList[x_, y_] := MemberQ[y, x];
Exist[f_] := ExistHolder[f];
ExistOuter[f_] := Hold[ToString[Evaluate[f][[0]]] != "Missing"];

CBackup := Module[{trash, name = "db/default-"<>ToString[dbbases["ver"]+1]},
    settings["backup date"] = Now;

    console["log", "backing up..."];
    Put[name<>".lock"];

    While[CreateDirectory[name] === $Failed,
      console["log", "looks like it already exists.. purging"];
      DeleteDirectory[name, DeleteContents->True];
      Pause[1];
    ];

    Do[ With[{j = i /. {Hold :> Unevaluated}}, DumpSave[name <> "/" <> ToString@Extract[i, 1, HoldForm] <> ".mx", j]; console["log", ">> dump ``", name <> ToString@Extract[i, 1, HoldForm] <> ".mx"] ], {i, Hold /@ Unevaluated[{ 
        experiments, 
        settings, 
        descriptions, 
        folders, 
        notebooks, 
        temp, 
        secretstring,
        collection,
        storage,
        CellObj
    }]}];

    console["log", "...done!"];

    DeleteFile[name<>".lock"];
    dbbases["ver"] += 1;
    dbbases["filename"] = name;

    If[Length@FileNames["*", "db"] > 100, 
        console["log", "autoclean"];
        trash = First@StringCases[#, 
            RegularExpression["((.*-)(\\d*))"] :> <|"filename" -> "$1", 
                "ver" -> ToExpression["$3"]|>] & /@ FileNames["*", "db"];

        trash = SortBy[trash, #["ver"] &];
        trash = Take[trash, (Length@FileNames["*", "db"] - 100)];
        DeleteDirectory[#["filename"], DeleteContents->True]&/@trash;
    ];

    PushNotification["DB", "A system backup was created"];
];

CXRule[x_, CXRule[op_, y_]] := op[x, y];
CXRule[x_String, CXRule[op_, y_]] := CXRule[#[x], CXRule[op, y]];

CApply[x_, op_] := CStamp[op, x];
CApply[x_String, y_] := CApply[#[x], y];
CApply[x_, CApply[y_, z_]] := CApply[CApply[x, y], z];
CApply[x_String, CRule[opA_, rest_]] := CPatter[x, opA, rest]
CPatter[x_, y_, z_] := CRule[CStamp[y, #[x]], z];

CXAnd[x__] := And[x];
CXAnd[x_List] := And @@ x;
CXOr[x__] := Or[x];
CXOr[x_List] := Or @@ x;
CXNot[x_] := Not[x];



CQuery[q_, lst_:"null"] := Module[{list, query},
   If[TrueQ[lst=="null"],list=collection//Keys;, list=lst];

  If[ListQ[q],

    If[Length[q] == 3 , list = folders[q[[3]], "list"]; ];

    query = q[[1]]//Evaluate;
    query = query /. {Not :> CNot, Or :> COr, And :> CAnd, RuleDelayed :> CApply, Rule -> CRule} /. {CRule :> CXRule};
    query = With[{f = query}, Function[f] ] /. {CAnd :> CXAnd, COr :> CXOr, CNot :> CXNot};

    query = query //. {CStamp[x_, y_] :> x[y]};

   

    With[{fun = query},
      Select[list, Function[x, fun[collection[x] ] ], q[[2]]]
    ]
  ,

    query = q//Evaluate;
    query = query /. {Not :> CNot, Or :> COr, And :> CAnd, RuleDelayed :> CApply, Rule -> CRule} /. {CRule :> CXRule};
    query = With[{f = query}, Function[f] ] /. {CAnd :> CXAnd, COr :> CXOr, CNot :> CXNot};

    query = query //. {CStamp[x_, y_] :> x[y]};

    
    
    With[{fun = query},
      Select[list, Function[x, fun[collection[x] ] ] ]
    ]
  ]
];

SortGroup[$$$s_, l_] :=
  If[Length[$$$s] == 0, l,
   With[{ccr = First[$$$s]}, Module[{groups = <||>, sortedkeys,function},
     With[{},
      function := ToExpression["Function[$, "<>ToString[ccr, InputForm]<>" // ReleaseHold // Evaluate // 
             ReleaseHold // Evaluate]"];

             

      With[{sym = ((function@collection[#]) /. {ExistHolder :> 
          ExistOuter} // ReleaseHold)/. {Missing :> Function[x, False]}},
          
          If[KeyExistsQ[groups, sym],
           groups[sym] = Append[groups[sym], #];
           ,
           groups[sym] = {#};
           ]
          
          ] & /@ l;
      ];
     
     
     Module[{group = KeySort[groups]},
      (group[#] = SortGroup[Drop[$$$s, 1], group[#]]) & /@ Keys[group];
      group
      ]
     ]
    ]
   ];

CSort[cri_, lst_] := 
  Module[{list, cr}, list = lst;
   $$$s = cri /. {RuleDelayed :> Rr} // Evaluate;
   $$$s = $$$s /. {Rr :> RuleDelayed};
   $$$s = {$$$s //. {RuleDelayed[x_, y_] :> y[$[x]]}} // Flatten;
   
   $$$s = If[StringQ[#], $[#], #] & /@ $$$s;
   
   SortGroup[$$$s, list]
   
   
   
   ];

CFlatten[l_] := l //. {Rule[x_, y_] :> y, Association :> List} // Flatten;

CGet := name |-> collection[name];
CPut := content |-> (collection[content["id"]]=KeyDrop[content, "data"]; collection[content["id"], "data"] = <||>; Map[(collection[content["id"], "data", #] = content["data", #])&, content["data"]//Keys]);

CKeyReset := (

  Clear[CKey];
  (*caching trick*)
  CKey[tag_][whatever_] := (With[{path = "assets/keys/"<>tag<>"/"<>whatever<>".wsp"}, If[FileExistsQ["public/"<>path]//TrueQ, CKey[tag][whatever] := LoadPage[path], With[{newpath = "assets/keys/"<>whatever<>".wsp"}, CKey[tag][whatever] := LoadPage[newpath] ] ] ]; CKey[tag][whatever]);

  CKey[tag_]["short"] := (With[{path = "assets/keys/"<>tag<>"/short.wls"}, If[FileExistsQ["public/"<>path]//TrueQ, CKey[tag]["short"] = Get["public/"<>path], With[{newpath = "assets/keys/short.wls"}, CKey[tag]["short"] = Get["public/"<>newpath] ] ] ]);

);

CKeyReset;