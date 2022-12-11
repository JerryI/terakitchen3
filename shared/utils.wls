NullQ[x_] := TrueQ[x == Null];

RandomString[n_] := Alphabet[][[RandomInteger[{1, 26}, n]]] // StringJoin;

(*misc*)
FilterRange[data_, r_] := 
  Select[data, (#[[1]] >= r[[1, 1]] && #[[1]] <= r[[1, 2]] && #[[2]] >=
        r[[2, 1]] && #[[2]] <= r[[2, 2]]) &];

DropHalf[x_] := Drop[x,-Length[x]/2 //Round];
		
lowpass[FilterEdge_, Q_, x_, sample_] := 
  1 - ((Tanh[(x - Length[sample[[All, 2]]]/2 + FilterEdge)/Q] + 1)/
     2 - (Tanh[(x - Length[sample[[All, 2]]]/2 - FilterEdge)/Q] + 1)/
     2);

ThicknessNormalise[target_][x_] := <|"cm"->1, "mm"->10, "um"->1/(0.0001)|>[target] Block[{cm=1, mm=0.1, um=0.0001}, ToExpression[x] ];