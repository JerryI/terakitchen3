input = $data["list"];

If[Length[input] == 0, PushNotification["Fourier","nothing to do..."]; $Die];

Module[{ref, sm, $sample, $ref, samplerate, fft, fftref,inc,freq, $samplethickness, leftcut, rightcut}, 

  console["log", "fourier for "<>#]; 
  sm = CGet[#]; 
  $sample = sm["data"]["tds"]; 
  
  If[OddQ[$sample//Length], $sample = Drop[$sample,-1]];
 
  samplerate = First[Round[1/Abs[Differences[$sample[[1 ;; 2]][[All,1]]]]]]; 
  inc = samplerate/Length[$sample]; 
  
  freq = Table[33.356*f, {f, 0, samplerate - inc, inc}]; 
  (*cut the unusable half*)
  freq = Drop[freq, -Length[freq]/2];
  (*leftcut = Position[freq, _?((Abs[# - cutoffL] < 1) &)] // First // First;
  rightcut = (Length[$sample]/2) - Position[freq, _?((Abs[# - cutoffR] < 1) &)] // First // First;*)
  
  (*logWrite["left cut" <> ToString[leftcut]];
  logWrite["right cut" <> ToString[rightcut]];*)
  
  fft = Fourier[$sample[[All,2]](*, FourierParameters -> {1, -1}*)]; 
  
  (*cut the unusable half*)
  fft = Drop[fft, -Length[fft]/2];
  sm["data"]["freqs"] = freq; 
  sm["data"]["fft"] = fft;  

  console["log", sm["id"]]; 
  CPut[sm];
] &  /@ input; 
          
PushNotification["Fourier", "Fourier transform is done"]; 

$Die;


