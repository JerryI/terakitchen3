NullQ[x_]:=TrueQ[x == Null];

CQuery[q_, lst_:"null"] := JTPClientEvaluate[kitchen, CQuery[q, lst]]; 

CEval[exp_] := JTPClientEvaluate[kitchen, exp];
SetAttributes[CEval, HoldAll];

CPut[item_] := JTPClientEvaluate[kitchen, CPut[item]]; 

CGet[id_] := JTPClientEvaluate[kitchen, collection[id]]; 
CGet[id_, field_] := JTPClientEvaluate[kitchen, collection[id][field]];
CGet[id_, field_, sub_] := JTPClientEvaluate[kitchen, collection[id, field, sub]];

PushNotification[x_,y_] := JTPClientEvaluate[kitchen, PushNotification[x,y]];

CreatePage[xs___] := JTPClientEvaluate[kitchen, CreatePage[xs]];
EventBind[type_, handler_] := (EventHandlers[type] = handler);
EmittedEvent[type_, cli_, data_] := Withh[{err=Catch[EventHandlers[type][cli, data]]}, If[StringQ[err], PushNotification["SVCore", StringTemplate["Failed. <i>``</i> Please, start over."][err]]; $Die]];

WebSocketSend[cli_, data_] := JTPClientEvaluate[kitchen, Tinyweb`WebSocketSend[cli, data]];
WebSocketPublish[exp_, chan_] := JTPClientEvaluate[kitchen, Tinyweb`WebSocketPublish[server, exp, chan]];

logs = {};

console["log", message_, args___] := (logs = {logs, StringTemplate[message][args]});

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
      {True, state}]], 
   "CloseFunction" -> Function[state, ClearAll[state]], 
   "WriteFunction" -> 
    Function[{state, bytes},(*Since we're writing to a cell,
     we don't want that trailing newline.*)
     With[{out = bytes /. {most___, 10} :> FromCharacterCode[{most}]},
       With[{ }, 
       If[out === "", {0, state},
        
        PushNotification["Debug", ByteArrayToString[out // ByteArray]];
        {Length@bytes, state}]]]]}
];

FilterRange[data_, r_] := 
  Select[data, (#[[1]] >= r[[1, 1]] && #[[1]] <= r[[1, 2]] && #[[2]] >=
        r[[2, 1]] && #[[2]] <= r[[2, 2]]) &];