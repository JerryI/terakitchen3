
__global__  void subtractFP (
    float4 * src, float * dest, mint len
) {
    const float4 params = src[1000];
    float* _dest = &dest[threadIdx.x << 3];

    const float thickness = params.y;

    float4 one    = src[threadIdx.x];

    //optimised using Experimental`OptimizedExpression on Wolfram Language
    const float n    = _dest[1];
    const float k    = _dest[2];

    const float freq = _dest[0];

    const float CONSTF = 6.28331f * thickness;

    float var40 = k*k;
    float var41 = 1.0f + n;
    float var54 = var41*var41;
    float var57 = var40 + var54;
    float var85 = n*n; 
    float var90 = 2.0f*CONSTF*freq*n;
    float var81 = -2.0f*CONSTF*freq*k;
    float var82 = expf(var81);
    float var58 = 1.0f/(var57*var57);
    float var93 = -1.0f + var40 + var85; 
    float var91 = cosf(var90);
    float var83 = -2.0f + k;
    float var84 = var83*k;
    float var86 = -1.0f + var84 + var85;
    float var87 = 2.0f + k;
    float var88 = k*var87;
    float var89 = -1.0f + var88 + var85;
    float var94 = sinf(var90);
    float var80 = var57*var57;
    float var99 = (var40 + (-1.0f + n)*(-1.0f + n));
    float var98 = expf(2.0f*CONSTF*freq*k);

    float var92 = 4.0f*k*var93*var94;

    float abs   = sqrtf(var58*(var99*var99/(var98*var98) + var80 + var82*(-2.0f*var86*var89*var91 + 2.0f*var92)));

    float arg = atan2f(var82*var58*(4.0f*k*var93*var91 + var86*var89*var94),var82*var58*(var98*var80 - var86*var89*var91 + var92));    

    if (!isfinite(abs) || !isfinite(arg)) {
        abs = 1.0f; arg = 0.0f;
    }

    if (abs < 0.0f) {abs = 1.0f; arg = 0.0f;};

    //debug
    _dest[5] = abs;
    _dest[6] = arg;

    _dest[3] = one.y * abs;
    _dest[4] = one.z - arg;
}

__global__ void solveNK (
    float4 * src, float * dest, mint len
) {
    const float4 params = src[1000];
    float* _dest = &dest[threadIdx.x << 3];

    const float n0 = params.x;
    const float thickness = params.y;

    //_dest[0] - freqs
    //_dest[1] - n
    //_dest[2] - k
    //_dest[3] - t abs
    //_dest[4] - t ph       
    
    const float fT = 1.0f/(thickness * _dest[0]);
    const float logT = logf(_dest[3]);
    const float ph   = _dest[4];

    float np = _dest[1];
    float kp = _dest[2];

    float n = 0;
    float k = 0;
    float denominator;
    float arg;
    float modulus;
    float im, re, n2;
    
    for(int i=0; i<30; ++i) {
        n2 = (1.0f + np);
        n2 = n2 * n2;
        denominator = 1.0f/(kp*kp + n2);
        denominator = denominator * denominator;

        re = denominator*(np*n2 + kp*kp*(2.0f + np));
        im = denominator*(kp*(kp*kp + np*np - 1.0f));

        modulus = sqrtf(re*re + im*im);
        arg = atan2f(im, re);

        n = 1.0f + 0.159152f * fT * (ph - arg);
        k = - 0.159152f * fT * (logT - logf(4.0f*modulus));

        np = n; kp = k;
    }

    if (!isfinite(n))
        n = n0;
    
    if (!isfinite(k))
        k = 0.0f;
    
    _dest[1] = n;
    _dest[2] = k;
}

__global__ void movingAverage (
    float4 * src, float * dest, mint len
) {
    float* _dest1 = &dest[(threadIdx.x << 2)];
    float* _dest2 = &dest[(threadIdx.x << 2) + (1 >> 3)];

    _dest2[1] = (_dest2[1] + _dest1[1])*0.5f;
    _dest2[2] = (_dest2[2] + _dest1[2])*0.5f;

    __syncthreads();

    _dest1[1] = (_dest2[1] + _dest1[1])*0.5f;
    _dest1[2] = (_dest2[2] + _dest1[2])*0.5f;
}

__global__ void initialise (
    float4 * src, float * dest, mint len
) {
    //__shared__ float4 transmission[1000];
    if (threadIdx.x > len) return;

    const float4 params = src[1000];
    float4 one    = src[threadIdx.x];
    //one.x - freq, 
    //one.y - abs, 
    //one.z - ph
    const float n0 = params.x;
    const float thickness = params.y;
    
    const float fT = 1.0f/(thickness * one.x);
    const float logT = logf(one.y);

    float* _dest = &dest[threadIdx.x << 3];

    float n = 1.0f + (0.159152f * (one.z + params.w) * fT);
    float k = - (0.159152f * logT * fT);

    if (!isfinite(n))
        n = n0;
    
    if (!isfinite(k))
        k = 0.0f;    

    _dest[0] = one.x;
    _dest[1] = n;
    _dest[2] = k;

    _dest[3] = one.y;
    _dest[4] = one.z + params.w;

}