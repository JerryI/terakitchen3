ClusterPhase = 
 Function[{input, low, high, PhaseTrashhold}, Module[{phase = input, delta = 0.0},
    
    Do[If[phase[[i + 1]] - phase[[i]] > PhaseTrashhold, 
       Do[phase[[j]] = phase[[j]] - 2 \[Pi];, {j, i + 1, 
          Length[phase]}];, 
       If[phase[[i + 1]] - phase[[i]] < -PhaseTrashhold, 
         Do[phase[[j]] = phase[[j]] + 2 \[Pi];, {j, i + 1, 
            Length[phase]}];];];
            
            , {i, low, high}];

    
    
    
    phase ] ];

