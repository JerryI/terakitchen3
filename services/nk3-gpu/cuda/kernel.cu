__device__ void movingAverage (
    float4 * src, float * dest
) {
    if (threadIdx.x > 1024 - 2) return;

    float* _dest = &dest[ threadIdx.x << 3];
    float* _next = &dest[(threadIdx.x + 1) << 3];

    _dest[1] = (_next[1] + _dest[1])/2.0f;
    _dest[2] = (_next[2] + _dest[2])/2.0f;
}

__device__  void subtractFP (
    float4 * src, float * dest
) {
    float4 params;
        params.x = dest[(1024 << 3)];     //n0
        params.y = dest[(1024 << 3) + 1]; //thickness
        params.z = dest[(1024 << 3) + 2]; //scale
        params.w = dest[(1024 << 3) + 3];
    //[0] - n0
    //[1] - thickness
    //[2] - scale
    //[3] - phase shift
    //
    float4 one = src[threadIdx.x];

    //scaling
    one.y = one.y * params.z;

    float* _dest = &dest[threadIdx.x << 3];

    const float thickness = params.y;

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

    if (one.y * abs > 1.1f) {
        _dest[3] = 1.0f;
        _dest[4] = 0.0f;
    } else {
        _dest[3] = one.y * abs;
        _dest[4] = one.z - arg + params.w;
    }
}

__device__ void solveNK (
    float4 * src, float * dest
) {
    float4 params;
        params.x = dest[(1024 << 3)];
        params.y = dest[(1024 << 3) + 1];
        params.z = dest[(1024 << 3) + 2];
        params.w = dest[(1024 << 3) + 3];

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

    if (!isfinite(n) || !isfinite(k) || n < 0.0f) {
        n = n0;
        k = 0.0f;
    }

    _dest[1] = n;
    _dest[2] = k;
}

__global__ void k_solveNK (
    float4 * src, float * dest
) {
    solveNK(src, dest);
}

__global__ void k_subtractFP (
    float4 * src, float * dest
) {
    subtractFP(src, dest);
}

__global__ void k_movingAverage (
    float4 * src, float * dest
) {
    movingAverage(src, dest);
}

__global__ void initialise (
    float4 * src, float * dest
) {
    //__shared__ float4 transmission[1000];
    if (threadIdx.x > 1023) return;

    const float4  params = src[1024];
    float4 one    = src[threadIdx.x];
    //one.x - freq, 
    //one.y - abs, 
    //one.z - ph
    const float n0 = params.x;
    const float thickness = params.y;
    
    const float fT = 1.0f/(thickness * one.x);
    const float logT = logf(one.y);

    float* _dest = &dest[threadIdx.x << 3];

    if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
        _dest[1024 << 3]     = params.x;
        _dest[(1024 << 3) + 1] = params.y;
        _dest[(1024 << 3) + 2] = params.z;
        _dest[(1024 << 3) + 3] = params.w;
    }

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

__global__ void generateTDS (
    float4 * src, 
    float * dest, 
    mint cycles_dry, 
    mint cycles_wet,
    float2 *dataset
) {

}    

__device__ void cpyDest (
    float * dest_local, float * dest
) {
    const int shift = blockIdx.x * (1025 * 8);

    dest[(threadIdx.x << 3) + shift] = dest_local[(threadIdx.x << 3)];
    dest[(threadIdx.x << 3) + shift + 1] = dest_local[(threadIdx.x << 3) + 1];
    dest[(threadIdx.x << 3) + shift + 2] = dest_local[(threadIdx.x << 3) + 2];
    dest[(threadIdx.x << 3) + shift + 3] = dest_local[(threadIdx.x << 3) + 3];
    dest[(threadIdx.x << 3) + shift + 4] = dest_local[(threadIdx.x << 3) + 4];
    dest[(threadIdx.x << 3) + shift + 5] = dest_local[(threadIdx.x << 3) + 5];
    dest[(threadIdx.x << 3) + shift + 6] = dest_local[(threadIdx.x << 3) + 6];
    dest[(threadIdx.x << 3) + shift + 7] = dest_local[(threadIdx.x << 3) + 7];    
}

__global__ void autorun (
    float4 * src, 
    float * dest, 
    mint cycles_dry, 
    mint cycles_wet,
    float3 *dataset
) {
    __shared__ float _local_dest[1024*8 + 8];

    const float4  params = src[1024];
    float4 one    = src[threadIdx.x];
    //one.x - freq, 
    //one.y - abs, 
    //one.z - ph
    const float n0 = params.x;
    const float thickness = dataset[blockIdx.x].x;
    const float scale = dataset[blockIdx.x].y;
    const float phase = dataset[blockIdx.x].z;
    
    const float fT = 1.0f/(thickness * one.x);
    const float logT = logf(one.y * scale);

    float* _dest = &_local_dest[threadIdx.x << 3];

    if (threadIdx.x == 0) {
        _dest[1024 << 3]       = params.x;
        _dest[(1024 << 3) + 1] = dataset[blockIdx.x].x;
        _dest[(1024 << 3) + 2] = dataset[blockIdx.x].y;
        _dest[(1024 << 3) + 3] = phase;
    }

    float n = 1.0f + (0.159152f * (one.z + phase) * fT);
    float k = - (0.159152f * logT * fT);

    if (!isfinite(n))
        n = n0;
    
    if (!isfinite(k))
        k = 0.0f;    

    _dest[0] = one.x;
    _dest[1] = n;
    _dest[2] = k;

    _dest[3] = one.y * scale;
    _dest[4] = one.z + phase;

    __syncthreads();

    solveNK(src, _local_dest);

    for (int i=0; i<cycles_dry; ++i) { 
      subtractFP(src, _local_dest);
      solveNK(src, _local_dest);
    }

    for (int i=0; i<cycles_wet; ++i) { 
      subtractFP(src, _local_dest);
      solveNK(src, _local_dest);
      __syncthreads();
      movingAverage(src, _local_dest);
    } 

    __syncthreads();
    cpyDest(_local_dest, dest);

}