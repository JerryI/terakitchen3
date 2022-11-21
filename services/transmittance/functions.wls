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





ExtractTransmittance[in1_, in2_, L_, trustreg_] := Module[
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




  transm = MapIndexed[{If[#2[[1]] > left && #2[[1]] < right, Abs[#1]^2, 1.0000001], 0} & , (nsam["fft"]/nref["fft"])];

  t0sam = (MaximalBy[nsam["tds"], Abs[#[[2]]] &] // First // First) - 
    nsam["tds"][[1, 1]];
  t0ref = (MaximalBy[nref["tds"], Abs[#[[2]]] &] // First // First) - 
    nref["tds"][[1, 1]];

  \[Phi]0sam = 2 \[Pi] (1/33.356) t0sam # & /@ nref["freqs"];
  \[Phi]0ref = 2 \[Pi] (1/33.356) t0ref # & /@ nref["freqs"];

  n0 = 1 + (Solve[10^-12 (t0sam - t0ref) == L/(2.9979 10^10) n1, n1] //
        Flatten)[[1, 2]];

  nsam["phase"] = (Arg[nsam["fft"]] - \[Phi]0sam);
  nref["phase"] = (Arg[nref["fft"]] - \[Phi]0ref);

  slog = nsam["phase"]; rlog = nref["phase"];

  transm[[All, 2]] = ClusterPhase[slog - rlog, If[extrapolatePhase == "yes", left, 1], right, PhaseTrashhold];
  slog = transm[[All, 2]];
 

  transm[[All, 2]] = transm[[All, 2]];

  transm[[All, 2]] = transm[[All, 2]] + \[Phi]0sam - \[Phi]0ref + phaseCorr;



  <|"transmission"->transm, "refractive index"->n0 |>
];
