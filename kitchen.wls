(*import shared libraries*)
Import/@FileNames["*.wls", "shared"];

(*load core modules*)
Import["utensils/db.wls"];
Import["utensils/misc.wls"];
Import["utensils/notifications.wls"];
Import["utensils/svhost.wls"];
Import["utensils/services.wls"];
Import["utensils/window.wls"];
Import["utensils/cells.wls"];
Import["utensils/notebook.wls"];

(*load libs from the services*)
Import/@FileNames["lib.wls", "services", 2];

(*check*)
If[FileExistsQ["running.lock"],
  console["log", ">>crashed before"];
  PushNotification["Server", "Wolfram Engine crashed. We are sorry &#128546;"];
];

(*clear pages*)
Put["running.lock"];

window = <||>;
temp = <||>;
settings["bucket"] = {};
settings["processes"] = <||>;
settings["cpu"] = 0;

(*debug stream*)
$Messages = {OpenWrite[Method -> "Toast"]};

(*async tasks*)

CronTasks["ping pong"] = SessionSubmit[
 ScheduledTask[WebSocketBroadcast[server, Pong[]], 
  Quantity[5, "Seconds"]]];

CronTasks["hide notifications"] = SessionSubmit[
 ScheduledTask[
(settings["notifications"][#]["hide"] = True) &/@ (Select[settings["notifications"]//Keys,  (settings["notifications", #, "date"] < (Now - settings["notifications", #, "duration"]))&]),
  Quantity[2, "Minutes"]]];

CronTasks["remove notifications"] = SessionSubmit[
 ScheduledTask[
(settings["notifications"][#] = .;)&/@ (Select[settings["notifications"]//Keys,  (settings["notifications"][#]["date"] < (Now - Quantity[1, "Days"]))&]),
  Quantity[6, "Hours"]]];  

CronTasks["storage date"] = SessionSubmit[
 ScheduledTask[SGlobalDate = Now; CArchive;,
  Quantity[12, "Hours"]]];  

CronTasks["backup"] = SessionSubmit[
 ScheduledTask[CBackup,
  Quantity[60, "Minutes"]]];

CronTasks["timestamp"] = SessionSubmit[
 ScheduledTask[console["log", "<*Now*>"],
  Quantity[10, "Minutes"]]]; 

(*
CronTasks["timestamp"] = SessionSubmit[
 ScheduledTask[PushNotification["Me", "Terakitchen betriebszeit erreicht 1 Tag &#127856;", "duration" -> Quantity[1, "Days"]],
  Quantity[1, "Days"]]];  *)

CronTasks["timestamp"] = SessionSubmit[
 ScheduledTask[PushNotification["Me", "Terakitchen betriebszeit erreicht 5 Tag &#127874;", "duration" -> Quantity[1, "Days"]],
  Quantity[5, "Days"]]];  

CronTasks["forecast"] = SessionSubmit[
 ScheduledTask[Forecast, "Daily"]];   
 
    

  

  

