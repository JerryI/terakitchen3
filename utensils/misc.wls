(*misc*)
FilterRange[data_, r_] := 
  Select[data, (#[[1]] >= r[[1, 1]] && #[[1]] <= r[[1, 2]] && #[[2]] >=
        r[[2, 1]] && #[[2]] <= r[[2, 2]]) &];
		
lowpass[FilterEdge_, Q_, x_, sample_] := 
  1 - ((Tanh[(x - Length[sample[[All, 2]]]/2 + FilterEdge)/Q] + 1)/
     2 - (Tanh[(x - Length[sample[[All, 2]]]/2 - FilterEdge)/Q] + 1)/
     2);

ShutDown := With[{}, 
    console["log", "stopping all tasks..."];
    TaskRemove/@Values@CronTasks["backup"]; 
    CBackup;
    DeleteFile["running.lock"];
    console["log", "bye"];
    Exit[];
];		

NullQ[x_] := TrueQ[x == Null];

$loop = <||>;
LoopSubmit[task_] := With[{uid = CreateUUID[]}, 

    $loop[uid] = <| "expression" -> Hold[task],
                    "time" -> Now,
                    "status" -> "running"
                |>;
  
    LoopTask[uid]
];

Unprotect[TaskAbort];
TaskAbort[LoopTask[uid_]] := ($loop[uid, "status"] = "aborted");

SetAttributes[LoopSubmit, HoldFirst];

DefineOutputStreamMethod[
  "DebugStream", {"ConstructorFunction" -> 
    Function[{name, isAppend, caller, opts}, 
     With[{state = Unique["JaBoo"]},
      {True, state}]], 
   "CloseFunction" -> Function[state, ClearAll[state]], 
   "WriteFunction" -> 
    Function[{state, bytes},(*Since we're writing to a cell,
     we don't want that trailing newline.*)
     With[{out = bytes /. {most___, 10} :> FromCharacterCode[{most}]},
       With[{ }, 
       If[out === "", {0, state},
        
        ByteArrayToString[out // ByteArray] >>> "warnings.log";
        console["log", ByteArrayToString[out // ByteArray]];

        {Length@bytes, state}]]]]}
];

Forecast := PushNotification["Weather", 
  StringDrop[(StringJoin@(StringTemplate["`` ```` C<br>"][
       CityData[#, "Name"], 
       With[{temp = 
          WeatherForecastData[#, "MaxTemperature"]["FirstValue"][[1]]},
        If[temp < 1, "&#10052;" , 
         If[temp > 25, "&#127774;", "&#9925;"]]
        ], 
       WeatherForecastData[#, "MaxTemperature"][
         "FirstValue"][[1]]] & /@ (GeoNearest[Entity["City"], Here, 
        2] // EntityList))), -4]
, "duration" -> Quantity[1, "Days"]];