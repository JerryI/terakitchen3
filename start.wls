#!/usr/bin/env wolframscript
(* ::Package:: *)

$env = <|
  "addr" -> "0.0.0.0",
  "port" -> 80,
  "path" -> "public",
  "storagehost" -> "tk.physik.uni-augsburg.de:82",
  "domain" -> "tk.physik.uni-augsburg.de"
|>;



(* ::Input:: *)
(**)


Print[">> INITIALIZATION STARTS"];

NullQ[x_]:=TrueQ[x == Null];

CronTasks = <||>;

Print[">> TERAKITCHEN V3 2022"];
Print[">> AUTOLAUNCH"];
console["log", message_, args___] := Print[StringTemplate[message][args]];

Do[

  console["log", ">> run: ``", i];
  Import[i];

, {i, SortBy[Select[FileNames["*.wls"], Function[x, x != "start.wls"]], If[StringContainsQ[#, "prior"], -1, 1]& ]}];


While[True,
    Pause[0.1];
    (Block[{This := $loop[#]}, $loop[#, "expression"]//ReleaseHold]; $loop[#] = .) &/@ Keys@$loop;
];