input = $data["list"];

If[Length[input] == 0, PushNotification["Rabi", "nothing to do..."]; $Die];


Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];

samples = CQuery[{"type"->Equal->"sample"}:>And, input];
console["log", "number: "<>ToString[samples//Length]];

fourier[data_] := With[{},
    tds = data;
    
    If[OddQ[tds//Length], tds = Drop[tds, -1]];
    
    samplerate = First[Round[1/Abs[Differences[tds[[1 ;; 2]][[All,1]]]]]]; 
    inc = samplerate/Length[tds]; 
    
    freq = Table[33.356*f, {f, 0, samplerate - inc, inc}]; 
    freq = Drop[freq, -Length[freq]/2];    
    fft = Fourier[tds[[All,2]]]; 

    (*cut the unusable half*)
    fft = Drop[fft, -Length[fft]/2]; 
];

startDialog[uuid_, ssam_, ssamid_] := (
    $uid = uuid;
    sam = ssam;
    samid = ssamid;

    $next = CreateUUID[];

    postRedirect[next_] := Function[data, If[data["action"] === "Ok", "window.wsp?id="<>next, "index.wsp"]];

    fourier[sam["tds"]];

    content =   Block[{}, 
                    LoadPage["services/rabi/views/range.wsp"]
                ];

    WebSocketPublish[LoadDynamicPage[content], $uid];            

    CreatePage[
        content,
        "id"           -> $uid,
        "postRedirect" -> postRedirect[$next],
        "postHandler"  -> ProcessEventEmitt[$id],
        "eventHandler" -> ProcessEventEmitt[$id, "event"]
    ];

    EventBind["submit", Function[{cli, data}, 
        If[data["action"] != "Ok", $Die];

        EventBind["event", Function[{cli, data}, Null;]];

        range = MinMax @ ImportString[data["points"], "JSON"];

        peaks = ToExpression /@ (StringReplace[#, {","->".", "\""->""}] & /@ {data["lpeak"], data["rpeak"]});
        peaks = Sort @ (ToExpression /@ peaks);

        If[!NumberQ[First@range] || !NumberQ[First@peaks] || !NumberQ[First@peaks],
            PushNotification["Rabi", StringTemplate["Wrong data was provided. Program was terminated. <i> `` or `` </i> is not a number"][range//Compress, peaks//Compress]];
            $Die;
        ];

        regions = Partition[
                Map[Function[x, (Position[sam["tds"][[All,1]], _?((Abs[# - x] < 1) &)] // First // First)], Flatten[range]]
            , 2] // Flatten;  
        fourier[sam["tds"][[regions[[1]] ;; regions[[2]]]]];


        $uid = $next;
        $next = CreateUUID[];

        prepareRabi;

        content =   Block[{}, 
                        LoadPage["services/rabi/views/fitting.wsp"]
                    ];

        WebSocketPublish[LoadDynamicPage[content], $uid];           

        If[Length[samples] > 0,
            postRedirect[next_] := Function[data, If[data["action"] === "Ok", "window.wsp?id="<>next, "index.wsp"]]; 
        ,
            postRedirect[next_] := Function[data, If[data["action"] === "Ok", "index.wsp", "index.wsp"]]; 
        ];
        
        CreatePage[
            content,
            "id"           -> $uid,
            "postRedirect" -> postRedirect[$next],
            "postHandler"  -> ProcessEventEmitt[$id],
            "eventHandler" -> ProcessEventEmitt[$id, "event"]
        ];

        PushNotification["Rabi", "Fitting procedure started <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];

        EventBind["submit", Function[{cli, data}, 
            If[data["action"] != "Ok", $Die]; 

            With[{f = freq, trace = simTDS, a = fft, id = samid, reg = regions, split = Abs[peaks[[1]]-peaks[[2]]], central = \[Omega]0, epsilon = \[Epsilon]\[Infinity], gamma = \[Gamma], strength = \[Omega]p},
                CEval[  
                    
                    collection[id, "data", "rabi-fft-abs"]  = Abs[a];
                    collection[id, "data", "rabi-fft-freq"] = f;
                    collection[id, "data", "rabi-reg"]  = reg;
                    collection[id, "data", "rabi-splitting"]  = split;

                    collection[id, "data", "rabi-fit-freq"]  = central;
                    collection[id, "data", "rabi-fit-epsilon"] = epsilon;
                    collection[id, "data", "rabi-fit-gamma"] = gamma;
                    collection[id, "data", "rabi-fit-strength"] = strength;

                    collection[id, "data", "rabi-fit-trace"] = trace;

                ]
            ];

            If[Length[samples] == 0, PushNotification["Rabi", "Done! Check the tickets to Stockholm &#127881;"]; $Die];

            startDialog[$next, CGet[samples//First, "data"], samples//First];
            samples = Drop[samples, 1];

            PushNotification["Rabi", "Don't miss some Rabi diamonds! &#9935; <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];            
        ]];  

        storage = ExportString[{{0},{0}}, "JSON"];    

        (*initial*)
        {l\[Epsilon], r\[Epsilon], l\[Omega], r\[Omega], l\[Gamma], r\[Gamma]} = {0.01, 20, 1, 20, 0.001, Abs[peaks[[2]]-peaks[[1]]]} // N; 

        EventBind["event", Function[{cli, data},
            Switch[data[[1]],
                "update",
                    {\[Epsilon]\[Infinity], \[Omega]p, \[Omega]0, \[Gamma]} = data[[2 ;; ]];

                    badvalue = False;
                    (If[!NumberQ[#], badvalue = #])&/@{\[Epsilon]\[Infinity], \[Omega]p, \[Omega]0, \[Gamma]};

                    If[badvalue === False,
                        Simulate;
                    ,
                        PushNotification["Rabi", StringTemplate["Invalid input parameters! ❌`` is not a number"][badvalue]];
                    ];

                    
                ,

                "more",
                    LoopSubmit[ fitRabi ]
                ,

                "correct",
                    LoopSubmit[ correctRabi ]
            ];        
        ]];




   
    ]];   

    EventBind["event", Function[{cli, data}, 
        range = MinMax @ ImportString[data, "JSON"];
        regions = Partition[
                Map[Function[x, (Position[sam["tds"][[All,1]], _?((Abs[# - x] < 1) &)] // First // First)], Flatten[range]]
            , 2] // Flatten;  
        fourier[sam["tds"][[regions[[1]] ;; regions[[2]]]]];  

        edges = Take[(SortBy[Transpose[{freq, PeakDetect[fft // Abs] fft // Abs}], Last] // Reverse), 2];      

        WebSocketSend[cli, UpdateGraph[ExportString[{freq, (fft//Abs)^2},"JSON"], ExportString[edges, "JSON"]] ];
    ]];


);


prepareRabi := (
   If[MissingQ[ref = CGet[CGet[samid, "ref"], "data"]],
        PushNotification["Rabi", StringTemplate["The problem with the reference of `` occured. Shutting down..."][samid]];
        $Die;
    ];

    If[MissingQ[thickness = CGet[samid, "thickness"]],
        thickness = With[{id = CGet[samid, "experiment"]}, CEval[experiments[id, "thickness"]]];
        If[MissingQ[thickness], PushNotification["Rabi", "No thickness was specified. Die."]; $Die];
    ];

    thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression@thickness]//N;
    If[!NumberQ[thickness], PushNotification["Rabi", StringTemplate["Thickness `` is not a number. Die."][thickness]]; $Die];


    t0sam = (MaximalBy[sam["tds"], Abs[#[[2]]] &] // First // First) - 
        sam["tds"][[1, 1]];
    t0ref = (MaximalBy[ref["tds"], Abs[#[[2]]] &] // First // First) - 
        ref["tds"][[1, 1]];

    \[Phi]0sam = 2 \[Pi] (1/33.356) t0sam # & /@ ref["freqs"];
    \[Phi]0ref = 2 \[Pi] (1/33.356) t0ref # & /@ ref["freqs"];

    originalTDS = If[OddQ[sam["tds"] // Length], Drop[sam["tds"], -1], sam["tds"]];


    n0 = 1 + (Solve[10^-12 (t0sam - t0ref) == thickness/(2.9979 10^10) n1,
           n1] // Flatten)[[1, 2]];
    
    semiClassical[\[Epsilon]\[Infinity]_, \[Omega]p_, \[Omega]0_, \
        \[Gamma]_] :=
          Table[
           With[{\[Epsilon] = \[Epsilon]\[Infinity] + \
        \[Omega]p^2/(\[Omega]0^2 - i^2 - I \[Gamma] i)},
            With[
             {L = thickness, \[Omega] = 
               2 \[Pi]  10^12 ((i/33.356)) , 
              c = 2.9979 10^10 , nair = 1,
              n = Re[Sqrt[\[Epsilon]]], k = Im[Sqrt[\[Epsilon]]]
              },

             (4 (n - I k) nair)/((n - I k) + 
                nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] 1/(
              1 - (((n - I k) - nair)/((n - I k) + 
                  nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])
             ]
            ], {i, sam["freqs"]}];   

    
    fastcmp[\[Epsilon]\[Infinity]_, \[Omega]p_, \[Gamma]_, \[Omega]0_] := (
       With[{sim = (Conjugate[
               With[{\[Epsilon] = \[Epsilon]\[Infinity] + \[Omega]p^2/(\
    \[Omega]0^2 - #^2 - I \[Gamma] #)},
                With[
                 {L = thickness , \[Omega] = 
                   2 \[Pi]  10^12  ((#/33.356)) , 
                  c = 2.9979 10^10 , nair = 1,
                  n = Re[Sqrt[\[Epsilon]]], k = Im[Sqrt[\[Epsilon]]]
                  },

                 (
                  4 (n - I k) nair)/((n - I k) + 
                    nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] 1/(
                  1 - (((n - I k) - nair)/((n - I k) + 
                      nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])
                 ]
                ]] & /@ ref["freqs"]) ref["fft"]},
        Abs[(Abs@(Fourier[originalTDS[[regions[[1]] ;; regions[[2]], 2]]]) - 
            Abs@(Fourier[(InverseFourier[
                  Join[sim, {0}, Drop[Reverse[Conjugate[sim]], -1]]] // 
                 Re)[[regions[[1]] ;; regions[[2]]]]]))]^2 // Total

        ]
   
   ); 


   (*setting up calculations*)

    {\[Epsilon]\[Infinity], \[Omega]p, \[Gamma]} = {n0^2, 1, 
      Abs[peaks[[2]]-peaks[[1]]]};

    \[Omega]0 = Mean@peaks;
    iterations = {};
    

    incr = samplerate/(regions[[2]] - regions[[1]]);

);

fitRabi := (

    If[MissingQ[ref = CGet[CGet[samid, "ref"], "data"]],
        PushNotification["Rabi", StringTemplate["The problem with the reference of `` occured. Shutting down..."][samid]];
        $Die;
    ];

    If[MissingQ[thickness = CGet[samid, "thickness"]],
        thickness = With[{id = CGet[samid, "experiment"]}, CEval[experiments[id, "thickness"]]];
        If[MissingQ[thickness], PushNotification["Rabi", "No thickness was specified. Die."]; $Die];
    ];

    thickness = Block[{cm = 1, mm = 0.1, um = 10^(-4)}, ToExpression@thickness]//N;
    If[!NumberQ[thickness], PushNotification["Rabi", StringTemplate["Thickness `` is not a number. Die."][thickness]]; $Die];


    t0sam = (MaximalBy[sam["tds"], Abs[#[[2]]] &] // First // First) - 
        sam["tds"][[1, 1]];
    t0ref = (MaximalBy[ref["tds"], Abs[#[[2]]] &] // First // First) - 
        ref["tds"][[1, 1]];

    \[Phi]0sam = 2 \[Pi] (1/33.356) t0sam # & /@ ref["freqs"];
    \[Phi]0ref = 2 \[Pi] (1/33.356) t0ref # & /@ ref["freqs"];

    originalTDS = If[OddQ[sam["tds"] // Length], Drop[sam["tds"], -1], sam["tds"]];


    n0 = 1 + (Solve[10^-12 (t0sam - t0ref) == thickness/(2.9979 10^10) n1,
           n1] // Flatten)[[1, 2]];
    
    semiClassical[\[Epsilon]\[Infinity]_, \[Omega]p_, \[Omega]0_, \
        \[Gamma]_] :=
          Table[
           With[{\[Epsilon] = \[Epsilon]\[Infinity] + \
        \[Omega]p^2/(\[Omega]0^2 - i^2 - I \[Gamma] i)},
            With[
             {L = thickness, \[Omega] = 
               2 \[Pi]  10^12 ((i/33.356)) , 
              c = 2.9979 10^10 , nair = 1,
              n = Re[Sqrt[\[Epsilon]]], k = Im[Sqrt[\[Epsilon]]]
              },

             (4 (n - I k) nair)/((n - I k) + 
                nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] 1/(
              1 - (((n - I k) - nair)/((n - I k) + 
                  nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])
             ]
            ], {i, sam["freqs"]}];   

    
    fastcmp[\[Epsilon]\[Infinity]_, \[Omega]p_, \[Gamma]_, \[Omega]0_] := (
       With[{sim = (Conjugate[
               With[{\[Epsilon] = \[Epsilon]\[Infinity] + \[Omega]p^2/(\
    \[Omega]0^2 - #^2 - I \[Gamma] #)},
                With[
                 {L = thickness , \[Omega] = 
                   2 \[Pi]  10^12  ((#/33.356)) , 
                  c = 2.9979 10^10 , nair = 1,
                  n = Re[Sqrt[\[Epsilon]]], k = Im[Sqrt[\[Epsilon]]]
                  },

                 (
                  4 (n - I k) nair)/((n - I k) + 
                    nair)^2 Exp[-I ((n - I k) - nair) (\[Omega] L)/c] 1/(
                  1 - (((n - I k) - nair)/((n - I k) + 
                      nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c])
                 ]
                ]] & /@ ref["freqs"]) ref["fft"]},
        Abs[(Abs@(Fourier[originalTDS[[regions[[1]] ;; regions[[2]], 2]]]) - 
            Abs@(Fourier[(InverseFourier[
                  Join[sim, {0}, Drop[Reverse[Conjugate[sim]], -1]]] // 
                 Re)[[regions[[1]] ;; regions[[2]]]]]))]^2 // Total

        ]
   
   ); 


   (*setting up calculations*)

    {\[Epsilon]\[Infinity], \[Omega]p, \[Gamma]} = {n0^2, 1, 
      Abs[peaks[[2]]-peaks[[1]]]};

    \[Omega]0 = Mean@peaks;
    iterations = {};
    

    incr = samplerate/(regions[[2]] - regions[[1]]);

    (*calc*)

    Do[
         iterations = {};
         Do[
          iterations = {iterations, {i, j, k, fastcmp[i, j, k, \[Omega]0]}};

          , {i, {l\[Epsilon], r\[Epsilon], (r\[Epsilon] - l\[Epsilon])/
            1000}}, {j, {l\[Omega], r\[Omega], (r\[Omega] - l\[Omega])/
            1000}}, {k, {l\[Gamma], r\[Gamma], (r\[Gamma] - l\[Gamma])/1000}}];

         {\[Epsilon]\[Infinity], \[Omega]p, \[Gamma], temp} = 
          First@MinimalBy[Partition[iterations // Flatten, 4], Last] // N;
         {l\[Epsilon], r\[Epsilon], l\[Omega], r\[Omega], l\[Gamma], 
           r\[Gamma]} = ({l\[Epsilon], r\[Epsilon], l\[Omega], r\[Omega], 
               l\[Gamma], 
               r\[Gamma]} + {\[Epsilon]\[Infinity], \[Epsilon]\[Infinity], \
        \[Omega]p, \[Omega]p, \[Gamma], \[Gamma]})/2 // N;

        Simulate;

     , {ll, 1, 10}];    

     PushNotification["Rabi", "Auto-search has finished"]; 

);

correctRabi := (

    iterations = {};
    

    incr = samplerate/(regions[[2]] - regions[[1]]);

    {l\[Epsilon], r\[Epsilon]} = \[Epsilon]\[Infinity] {0.9, 1.1};
    {l\[Omega], r\[Omega]} = \[Omega]p {0.9, 1.1};
    {l\[Gamma], r\[Gamma]} = \[Gamma] {0.9,1.1};

    (*calc*)

    Do[
         iterations = {};
         Do[
          iterations = {iterations, {i, j, k, fastcmp[i, j, k, \[Omega]0]}};

          , {i, {l\[Epsilon], r\[Epsilon], (r\[Epsilon] - l\[Epsilon])/
            1000}}, {j, {l\[Omega], r\[Omega], (r\[Omega] - l\[Omega])/
            1000}}, {k, {l\[Gamma], r\[Gamma], (r\[Gamma] - l\[Gamma])/1000}}];

         {\[Epsilon]\[Infinity], \[Omega]p, \[Gamma], temp} = 
          First@MinimalBy[Partition[iterations // Flatten, 4], Last] // N;
         {l\[Epsilon], r\[Epsilon], l\[Omega], r\[Omega], l\[Gamma], 
           r\[Gamma]} = ({l\[Epsilon], r\[Epsilon], l\[Omega], r\[Omega], 
               l\[Gamma], 
               r\[Gamma]} + {\[Epsilon]\[Infinity], \[Epsilon]\[Infinity], \
        \[Omega]p, \[Omega]p, \[Gamma], \[Gamma]})/2 // N;

        Simulate;

     , {ll, 1, 10}];    

     PushNotification["Rabi", "Correcting has finished"]; 

);

Simulate := (
    With[{sim = 
        Conjugate[
            semiClassical @@ {\[Epsilon]\[Infinity], \[Omega]p, 
                \[Omega]0, \[Gamma] // Abs}] ref["fft"]},

        simTDS = InverseFourier[Join[sim, {0}, Drop[Reverse[Conjugate[sim]], -1]]] // Re;

        With[{r = {originalTDS[[All, 1]], simTDS}},
            storage = ExportString[r, "JSON"];
            jvars = ExportString[{\[Epsilon]\[Infinity], \[Omega]p, \[Omega]0, \[Gamma]}, "JSON"];

            WebSocketPublish[UpdateGraph[storage, jvars], $uid];
        ];
        

    ];
 
);


startDialog[CreateUUID[], CGet[samples//First, "data"], samples//First];
samples = Drop[samples, 1];

PushNotification["Rabi", "Let's dig for some Rabi &#9935; <br>"<>"<a class=\"btn btn-outline-info\" href=\"window.wsp?id="<>$uid<>"\">open</a>"];