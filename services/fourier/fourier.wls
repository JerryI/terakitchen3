input = $data["list"];

If[Length[input] == 0, PushNotification["services/fourier","nothing to do..."]; $Die];

FFT := (
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
);

NUFFT := (
    data = $sample;
    nUniquePts = Ceiling[(Length[data]+1)/2];
    {t0, t1} = #[[1]] & /@ data[[{1, -1}]];

    delta = t1 - t0;

    nPoints = Length@data;

    ifun = Interpolation[data];

    sample = Table[{t0 + i/delta, ifun[t0 + i/nPoints*delta]}, {i, 0, nPoints - 1}];

    fft = Fourier@(Transpose@sample)[[2]];
    freq = (33.356) N@(Range[0, nPoints - 1]/delta);

    sm["data"]["freqs"] = Take[freq,   nUniquePts]; 
    sm["data"]["fft"] = Take[fft,   nUniquePts];  
);

Module[{}, 

  console["log", "fourier for "<>#]; 
  sm = CGet[#]; 
  $sample = sm["data"]["tds"]; 
  
  If[OddQ[$sample//Length], $sample = Drop[$sample,-1]];


 
  If[interpolate === True,
    NUFFT;
  ,
    (*check consistensy*)
    If[(std = StandardDeviation@Differences@($sample[[All,1]])) > 0.00008,
      With[{plot = ExportString[ListLinePlot[Differences[$sample[[All,1]]], AxesLabel->"ns", ImageSize->Small, AspectRatio->1.2],"SVG"] },
        PushNotification["services/fourier", StringTemplate["Irregular samplerate was detected in <a href=\"item/index.wsp?id=``\" class=\"badge badge-light\">``</a>. <br> <br> `` <br> The standart deviation is <span class=\"badge badge-info\">``E5</span>. You shoud consider to rerun it with an <a class=\"badge badge-info\" href=\"service.wsp?id=oversampler\">oversampler</a>"][#,#,plot, Round[std,0.0000001]100000 ] ];
      ];
      If[interpolate === True, NUFFT, FFT];
    ,
      FFT
    ];    
  ];

  console["log", sm["id"]]; 
  With[{uid = sm["id"], fft = sm["data", "fft"], freqs = sm["data"]["freqs"], len = Floor[Length[sm["data", "fft"] ]/2 ]},
    CEval[
      collection[uid, "data", "fft"] = fft;
      collection[uid, "attributes", "usable-range"] = Sequence[;; len];
      collection[uid, "data", "freqs"] = freqs;
      collection[uid, "data", "fft-abs"] = With[{id=uid},collection[id,"data","fft"]//Abs];
    ]
  ];
] &  /@ input; 
          
PushNotification["services/fourier", "Fourier transform is done"]; 

$Die;


