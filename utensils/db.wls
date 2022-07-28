(*loading DB*)

dbbases = 
 First@StringCases[#, 
     RegularExpression["((.*-)(\\d*))"] :> <|"filename" -> "$1", 
       "ver" -> ToExpression["$3"]|>] & /@ FileNames["*", "db"];

If[Length@dbbases == 0, console["log", "no db files was found. Abort"]; Exit[]];

dbbases = SortBy[dbbases, -#["ver"] &];

While[TrueQ[FileExistsQ[(First@dbbases)["filename"] <> ".lock"]],
  DeleteFile[(First@dbbases)["filename"] <> ".lock"];
  console["log", "db `` crashed. skipping...", (First@dbbases)["filename"] ];
  dbbases = Drop[dbbases, 1];
];

dbbases = First@dbbases;
console["log", "openning ``", dbbases ];

Get[dbbases["filename"]];

(*db operations*)
Exist[f_] := ExistHolder[f];
ExistOuter[f_] := Hold[ToString[Evaluate[f][[0]]] != "Missing"];

CBackup := Module[{trash, name = "db/default-"<>ToString[dbbases["ver"]+1]},
    console["log", "backing up..."];
    Put[name<>".lock"];

    DumpSave[name, {collection, experiments, settings, descriptions, folders}];
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
        DeleteFile[#["filename"]]&/@trash;
    ];
];

CQuery[q_, lst_:"null"] := Module[{list},
   If[TrueQ[lst=="null"],list=collection//Keys;, list=lst];

   $$$q = q /. {RuleDelayed :> Rr} // Evaluate;
   $$$q = $$$q /. {Rr :> RuleDelayed};
   $$$q = $$$q //. {RuleDelayed[x_, y_] :> 
       If[! ListQ[x], 
        Rule[y[[1]][If[StringQ[x], Hold[$[x]], x]], y[[2]]], y @@ x], 
      Rule[x_, y_] :> 
       If[ToString[y[[2, 0]]] == "List", y[[1]] @@ y[[2]], 
        y[[1]][If[StringQ[x], Hold[$[x]], x], y[[2]]]]};
   
    Select[
        list
     ,
     Function[$$, (Function[$, $$$q // ReleaseHold // Evaluate // 
             ReleaseHold // Evaluate]@collection[$$]) /. {ExistHolder :> 
          ExistOuter} // ReleaseHold]
     ]
];

SortGroup[$$$s_, l_] :=
  If[Length[$$$s] == 0, l,
   With[{ccr = First[$$$s]}, Module[{groups = <||>, sortedkeys,function},
     With[{},
      function := ToExpression["Function[$, "<>ToString[ccr, InputForm]<>" // ReleaseHold // Evaluate // 
             ReleaseHold // Evaluate]"];

             

      With[{sym = (function@collection[#]) /. {ExistHolder :> 
          ExistOuter} // ReleaseHold},
          
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
CPut := content |-> collection[content["id"]]=content;