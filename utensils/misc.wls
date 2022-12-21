Print[">> Shared Libs V1"];

ShutDown := With[{}, 
    console["log", "stopping all tasks..."];
    PushNotification["Cron", "Server was down"];
    TaskRemove/@Values@CronTasks; 
    CBackup;
    DeleteFile["running.lock"];
    console["log", "bye"];
    Exit[-1];
];	

console["memory stat"] := (
  settings["memory"]={settings["memory"], Round[MemoryInUse[]/1024,1]/1024//N}//Flatten;
);



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
  "Toast", {"ConstructorFunction" -> 
    Function[{name, isAppend, caller, opts}, 
     With[{state = Unique["JaBoo"]},
      {True, state}] ], 
   "CloseFunction" -> Function[state, ClearAll[state] ], 
   "WriteFunction" -> 
    Function[{state, bytes},(*Since we're writing to a cell,
     we don't want that trailing newline.*)
     With[{out = bytes /. {most___, 10} :> FromCharacterCode[{most}]},
       With[{ }, 
       If[out === "", {0, state},
        
        PushNotification["system", "<span style=\"color:red\">"<>ByteArrayToString[out // ByteArray]<>"</span>"];
        StringTemplate["<*Now*>"][] >>> "warnings.log";
        ByteArrayToString[out // ByteArray] >>> "warnings.log";

        {Length@bytes, state}] ] ] ]}
];

Forecast := PushNotification["Weather", 
  StringDrop[(StringJoin@(StringTemplate["`` ```` C<br>"][
       CityData[#, "Name"], 
       With[{temp = 
          WeatherForecastData[#, "MaxTemperature"]["FirstValue"][[1]]},
        If[temp < 1, "&#10052;", 
         If[temp > 25, "&#127774;", "&#9925;"]]
        ], 
       WeatherForecastData[#, "MaxTemperature"][
         "FirstValue"][[1]]] & /@ (GeoNearest[Entity["City"], Here, 
        2] // EntityList))), -4]
, "duration" -> Quantity[1, "Days"]];

AnalyseTags[last_:200] := Module[{keys},
	keys = Take[collection//Keys, -last];
	
	Map[Function[key,
	  {
		Select[
			Complement[(collection[key]//Keys), 
			{"id", "data", "date", "ref", "descid", "experiment"}]
		, Function[tag, BooleanQ[collection[key, tag]]]]
	  , collection[key, "experiment"]}
	], keys]//Flatten
];

TagCloud := 
 Module[{uid = CreateUUID[], url = ""}, 

  Export[StringJoin["public/trashcan/", uid, ".svg"], 
    WordCloud[AnalyseTags[200], 
     Rectangle @@ ({{0, 0}, {3, 1.5}} - Mean[{{0, 0}, {3, 1.5}}]), 
     ImageSize -> Small] ]; 

  url = StringJoin["http://", $env["storagehost"], "/trashcan/", uid, 
    ".svg"]; 
  PushNotification["Recent", StringJoin["<img src=\"",url,"\">"], 
   "duration" -> Quantity[12, "Hours"] ] ];