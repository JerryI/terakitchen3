DifferencesNormalized[arr_] := Differences[arr];

ClusterPhase = 
  Compile[{{input, _Real, 1}, {low, _Integer, 0}, {high, _Integer, 
     0}, {PhaseTrashhold, _Real, 0}}, Module[{phase = input, delta = 0.0},
    
    Do[If[phase[[i + 1]] - phase[[i]] > PhaseTrashhold, 
       Do[phase[[j]] = phase[[j]] - 2 \[Pi];, {j, i + 1, 
          Length[phase]}];, 
       If[phase[[i + 1]] - phase[[i]] < -PhaseTrashhold, 
         Do[phase[[j]] = phase[[j]] + 2 \[Pi];, {j, i + 1, 
            Length[phase]}];];];, {i, low, high}];
    If[low > 1,
    
        delta=Differences[phase[[low;;Floor[(3 high+low)/4]]]]//Mean;
    
        Do[phase[[-i-1]]=phase[[-i]]-delta/2;,{i,-low,-2}];
    ];
    
    
    delta = 
     Differences[phase[[Floor[(high + 3 low)/4] ;; high]]] // Mean;
    
    Do[phase[[i + 1]] = phase[[i]] + delta;, {i, high, 
      Length[phase] - 1}];
    
    
    
    phase], CompilationTarget -> "C"];





(*-------fixed point algorithm--------*)
(*doi:10.1117/12.612946*)
(*------------------------------------*)

(*precompile if it is possible*)
NK := Function[{freqs, Tfft, L, left, right, n00}, 
   Module[{phasediff, n0 = 0.0, k0 = 0.0, n = 0.0, k = 0.0, b = 1},
    
    Table[
     Check[
      If[left > i || i > right,
       
       n0 = n00;
       
       
       k0 = 0;,
       n0 = (2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
               33.356) L) (Tfft[[i, 2]]) + 1;
       k0 = -((2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
                33.356) L)) (Log[Tfft[[i, 1]]]);
       Do[
        n = (2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
                33.356) L) ((Tfft[[i, 2]]) - 
             Arg[((4 (n0 - I k0))/((n0 - I k0) + 1)^2)]) + 1;
        k = -((2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
                 33.356) L)) (Log[Tfft[[i, 1]]] - 
            Log[Abs[(4 (n0 - I k0))/((n0 - I k0) + 1)^2]]);
        
        n0 = n; k0 = k;
        
        
        , {j, iterations}];
        
       
       
       ];
      {n0, k0}
      
      , {n00, 0}],
     {i, 1, Length[freqs]}]]];

NK2Continue := 
  Function[{freqs, Tfft, L, left, right, nini, kini}, 
   Module[{phasediff = 0.0, n0 = 0.0, k0 = 0.0, n = 0.0, k = 0.0}, 
    Table[n0 = nini[[i]];
     k0 = kini[[i]];
     If[left > i || i > right, {n0, k0},
      
      Check[
       Do[
        n = (2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
                33.356) L) ((Tfft[[i, 2]]) - 
             Arg[(4 (n0 - I k0))/((n0 - I k0) + 1)^2]) + 1;
        k = -((2.9979 10^10)/(2 \[Pi] 10^12 (freqs[[i]]/
                 33.356) L)) (Log[Tfft[[i, 1]]] - 
            Log[Abs[(4 (n0 - I k0))/((n0 - I k0) + 1)^2]]);
        n0 = n; k0 = k;
        
        , {j, iterations}];
        
       {n, k}, {n0, k0}]
      ], {i, 1, Length[freqs]}]]];


SubtractFPT[nk_, tr_, Ll_, left_, right_] := 
  Module[{FP}, 
   Table[With[{L = Ll "cm", \[Omega] = 
       2 \[Pi] 10^12 ("sec")^-1 ((nk[[i, 1]])/33.356), 
      c = 2.9979 10^10 ("cm")/("sec"), nair = 1, n = nk[[i, 2, 1]], 
      k = nk[[i, 2, 2]], \[Delta] = 30}, 
     FP = Conjugate[(1/(1 - (((n - I k) - nair)/((n - I k) + 
                 nair))^2 Exp[-2 I (n - I k) (\[Omega] L)/c]))];
     
     Check[If[i < left || i > right,
       {tr[[i, 1]], tr[[i, 2]]},
       {tr[[i, 1]]/Abs[FP], tr[[i, 2]] - Arg[FP]}
       ], {tr[[i, 1]], tr[[i, 2]]}]
     
     ], {i, 1, tr // Length}]];

TDSwithoutFP[in1_, in2_, L_, trustreg_, checkregions_, first_] := 
 Module[{nsam = in1, \[Phi]0sam, \[Phi]0ref, slog, nref = in2, 
   iterations, errors, backup, transm, variants = {}, regions = {},simulated, 
   t0sam, t0ref, n0, left, right}, 
  left = Position[nsam["freqs"], _?((Abs[# - trustreg[[1]]] < 1) &)] //
      First // First;
  right = 
   Position[nsam["freqs"], _?((Abs[# - trustreg[[2]]] < 1) &)] // 
     First // First;


  transm = {Abs[#], 0} & /@ (nsam["fft"]/nref["fft"]);


  t0sam = (MaximalBy[nsam["tds"], Abs[#[[2]]] &] // First // First) - 
    nsam["tds"][[1, 1]];
  t0ref = (MaximalBy[nref["tds"], Abs[#[[2]]] &] // First // First) - 
    nref["tds"][[1, 1]];

  \[Phi]0sam = 2 \[Pi] (1/33.356) t0sam # & /@ nref["freqs"];
  \[Phi]0ref = 2 \[Pi] (1/33.356) t0ref # & /@ nref["freqs"];


  nsam["phase"] = (Arg[nsam["fft"]] - \[Phi]0sam);
  nref["phase"] = (Arg[nref["fft"]] - \[Phi]0ref);

  slog = nsam["phase"]; rlog = nref["phase"];

  transm[[All, 2]] = ClusterPhase[slog - rlog, If[extrapolatePhase == "yes", left, 1], right, PhaseTrashhold];
  slog = transm[[All, 2]];

  transm[[All, 2]] = transm[[All, 2]];

  transm[[All, 2]] = transm[[All, 2]] + \[Phi]0sam - \[Phi]0ref + phaseCorr;
  transm = SubtractFPT[first, transm, L, left, right];
  transm = (#[[1]] Exp[I #[[2]] ]) & /@ transm;
  simulated = With[{sim = nref["fft"] transm},
        InverseFourier[Join[sim, {0}, Drop[Reverse[Conjugate[sim]], -1]]] // Re
  ];
  
  simulated
];

fftOsc[freqs_, data_] := 
 Module[{$sample = data, samplerate, inc, freq, fft, cut},
  
  
  samplerate = First[Round[1/Abs[Differences[freqs[[1 ;; 2]]]]]];
  inc = samplerate/Length[freqs];
  
  freq = Table[f, {f, 0, samplerate - inc, inc}];

  cut = -Floor[Length[freq]/2];
  (*cut the unusable half*)
  freq = Drop[freq, cut];
  
  
  fft = Fourier[$sample(*,FourierParameters->{1,-1}*)] // Abs;
  
  (*cut the unusable half*)
  {freq, Drop[fft, cut]} // Transpose
];



estimateOscillations[f_, FPperiod_] := 
  With[{piece = {#[[1]], ((#[[2, 
              2]] 4 \[Pi] 10^12 #[[1]])/(33.356 2.9979 10^10))} & /@ 
      f},
  
   With[{spectrum = (fftOsc @@ 
         Transpose@
          piece)}, 
          
          With[{filtered = FilterRange[spectrum, {{0.9/(2 FPperiod), 9999}, {-9999, 9999}}] },
            Function[x, (x[[2, 1]] - x[[1, 1]]) Total[x[[All, 2]]]// N] @ filtered
          ]
  ]

];

ExtractNK[in1_, in2_, L_, trustreg_, checkregions_] := Module[
    {
        nsam = in1, \[Phi]0sam, \[Phi]0ref, slog, nref = in2, first, 
   iterations, errors, backup, transm, variants = {}, regions = {}, 
   t0sam, t0ref, n0, left, right, FPperiodIndexes, FPperiod},

   

  left = Position[nsam["freqs"], _?((Abs[# - trustreg[[1]]] < 1) &)] //
      First // First;
  right = 
   Position[nsam["freqs"], _?((Abs[# - trustreg[[2]]] < 1) &)] // 
     First // First;
    
  nsam["fft"] = nsam["fft"] amplify;




  transm = {Abs[#], 0} & /@ (nsam["fft"]/nref["fft"]);

  


  t0sam = (MaximalBy[nsam["tds"], Abs[#[[2]]] &] // First // First) - 
    nsam["tds"][[1, 1]];
  t0ref = (MaximalBy[nref["tds"], Abs[#[[2]]] &] // First // First) - 
    nref["tds"][[1, 1]];

  \[Phi]0sam = 2 \[Pi] (1/33.356) t0sam # & /@ nref["freqs"];
  \[Phi]0ref = 2 \[Pi] (1/33.356) t0ref # & /@ nref["freqs"];

  n0 = 1 + (Solve[10^-12 (t0sam - t0ref) == L/(2.9979 10^10) n1, n1] //
        Flatten)[[1, 2]];

  FPperiod = With[
     {LL = L "cm", 
      c = 2.9979 10^10 ("cm")/("sec"), nair = 1
      },
      2 Pi/(4 n0 (2 \[Pi]  10^12 ("sec")^-1 ((1/33.356)) LL)/c)
     ];           



  nsam["phase"] = (Arg[nsam["fft"]] - \[Phi]0sam);
  nref["phase"] = (Arg[nref["fft"]] - \[Phi]0ref);

  slog = nsam["phase"]; rlog = nref["phase"];

  transm[[All, 2]] = ClusterPhase[slog - rlog, If[extrapolatePhase == "yes", left, 1], right, PhaseTrashhold];
  slog = transm[[All, 2]];
 

  transm[[All, 2]] = transm[[All, 2]];

  transm[[All, 2]] = transm[[All, 2]] + \[Phi]0sam - \[Phi]0ref + phaseCorr;

  first = 
   Transpose[{nsam["freqs"], 
     NK[nsam["freqs"], transm, L, left, right, n0]}];

  first[[All, 2, 1]] = 
   Join[{first[[1, 2, 1]]}, MovingAverage[first[[All, 2, 1]], 2]];
  first[[All, 2, 2]] = 
   Join[{first[[1, 2, 2]]}, MovingAverage[first[[All, 2, 2]], 2]];

  backup = transm;
  iterations = {};
  errors = {};

  regions = Partition[
  
    Map[Function[
      x, (Position[nsam["freqs"], _?((Abs[# - x] < 1) &)] // 
         First // First)], Flatten[checkregions]]
     
     , 2];


    
    appendVariant[interation_] := (
    
      With[{err = (estimateOscillations[first[[ #[[1]] ;; #[[2]] ]], FPperiod] &/@ regions )},
       
        variants = {variants, <|"thickness" -> L, "first" -> first, "iteration" -> interation, "amplitude" -> amplify,
                              "error" ->  Total[err],
                              "sampling"-> {}
       
       |>};

       

 
       
      ];
      
    );


    appendVariant[0];
  
    
    Do[transm = SubtractFPT[first, backup, L, left, right];
   
     
     (*transm[[All,2]]=ClusterPhase[transm[[All,2]],left,right];*)
     (*transm[[All,2]]=transm[[All,2]]-0 2\[Pi](Floor[Fit[transm[[All,
     2]],{1,x},x][[1]]/(2\[Pi])]);*)
     first = Transpose[{nsam["freqs"], 
         NK2Continue[nsam["freqs"], transm, L, left, right, 
          first[[All, 2, 1]], first[[All, 2, 2]]]}] // N;
     first[[All, 2, 1]] = If[# < -0.1, 0, #]&/@first[[All, 2, 1]];   
     first[[All, 2, 2]] = If[# < -0.1, 0, #]&/@first[[All, 2, 2]];  
     first[[All, 2, 1]] = 
      Join[{first[[1, 2, 1]]}, MovingAverage[first[[All, 2, 1]], 2]];
     first[[All, 2, 2]] = 
      Join[{first[[1, 2, 2]]}, MovingAverage[first[[All, 2, 2]], 2]];
    
    appendVariant[i];
     
     
     , {i, 1, numberFP}];
    (*transm[[All,2]]=transm[[All,2]]-0 2\[Pi](Floor[Fit[transm[[All,
    2]],{1,x},x][[1]]/(2\[Pi])]);*)
     {variants//Flatten, first}
    ];

ProcessNKQuet[calculated_] := With[
    {
      
    },
    
    (*add points to the plot*)
    (
        (*add to the big collection*)
        variants = {variants, #}; 
        
    ) & /@ calculated;  
];    

ProcessNK[calculated_] := With[
    {
      
    },
    
    (*add points to the plot*)
    (
        (*add to the big collection*)
        variants = {variants, #}; 
        
        (*generate color*)
        With[{r =   {
                        #["thickness"], #["error"], 
                        With[{col = Blend[{{0, Blue//Lighter},{numberFP/2, Red}, {numberFP, Green}}, #["iteration"]]}, 
                            StringTemplate["rgb(``,``,``,64)"][Round[col[[1]] 255, 1], Round[col[[2]] 255, 1], Round[col[[3]] 255, 1]]
                        ]
                    }
             },
            (*publish*)
           
            With[{rr = ExportString[r, "JSON"], id = $uid}, 
                WebSocketPublish[UpdateGraph[rr], $uid];
                stack = {stack, rr};
            ];
        ];
    ) & /@ calculated;  
    (*update sampling graph*)
    With[{r = {#["first"][[All,1]], Function[x,((x[[2, 2]] 4 \[Pi] 10^12 x[[1]])/(33.356 2.9979 10^10))]/@(#["first"]), #["sampling"]}},
        WebSocketPublish[UpdateSamplingGraph[ExportString[r, "JSON"]], $uid];        
    ]&@Last[calculated];
];    