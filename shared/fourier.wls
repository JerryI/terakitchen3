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