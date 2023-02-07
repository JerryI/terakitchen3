input = $data["list"];
$uid  = CreateUUID[];

phcorr = 0.0;

Needs["CUDALink`"];
If[!CUDAQ[], PushNotification["services/nk3-gpu","<span class=\"badge badge-danger\">GPU CUDA is not supported</span>"]; $Die];

If[Length[input] == 0, PushNotification["services/nk3-gpu","nothing to do..."]; $Die];

(* CUDA *)
autorun = CUDAFunctionLoad[Import["services/nk3-gpu/cuda/kernel.cu", "String"], "autorun", {{"Float[4]", "Input"}, {"Float", "Output"}, _Integer,  _Integer, {"Float[2]", "Input"}}, 1024, "CleanIntermediate"->False];
If[Head[autorun] =!= CUDAFunction, PushNotification["services/nk3-gpu","<span class=\"badge badge-danger\">Could not compile the kernel</span>"]; $Die]

Import["https://raw.githubusercontent.com/JerryI/tinyweb-mathematica/master/WSP/WSP.wl"];
Import["services/nk3-gpu/windows.wls"];
Import["services/nk3-gpu/functions.wls"];

samples = CQuery[{"type"->Equal->"sample"}:>And, input];

openDialogWindow["Check the parameters"][samples//First];
samples = Drop[samples, 1];
