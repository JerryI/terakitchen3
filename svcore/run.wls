Import["https://raw.githubusercontent.com/JerryI/tcp-mathematica/main/JTP/JTP.wl"];
(*import shared libraries*)
Import/@FileNames["*.wls", "shared"];

(*only for the SVCORE*)
Import["svcore/lib.wls"];

kitchen = (JTPClient@@Uncompress[$kitchen]) // JTPClientStart;

eventlistener = JTPServer[] // JTPServerStart;
eventlistener["promise"] = Null;

$data = With[{uid = $id, eengine = List["host" -> eventlistener["host"], "port" -> eventlistener["port"]]},
    JTPClientEvaluate[kitchen, ProcessCheckIn[uid, eengine]]
];

$Die := With[{uid = $id, l = Flatten[logs]},
    JTPClientEvaluate[kitchen, ProcessCheckOut[uid, l]];
    Exit[];
];

$Messages = {OpenWrite[Method -> "Toast"]};

With[{symbol = Symbol[#["symbol"]], val = #["value"]}, Set[symbol, val]] &/@ $data["init"];

Import[$data["task"]];

While[True, 
    (Block[{This := $loop[#]}, $loop[#, "expression"]//ReleaseHold]; $loop[#] = .) &/@ Keys@$loop;
    Pause[0.1];
];





