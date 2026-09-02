// ggml-cuda8-mulmat-f16-smoke.cu
//
// G49 (increment 2): correctness gate + perf check for the F16-src0 attention
// matmul (ggml_cuda8_mul_mat_f16_f32_launch). Proves the F16 READ path before
// any dispatch wiring, and reports ms/call so the convert-in-loop cost can be
// compared against the F32 variant's 0.13 ms/call.
//
// Standalone .cu: launcher declared extern "C"; all tensors built by hand.
// Uses <stdint.h> (not <cstdint>) to avoid the GCC 5.4 C++11 library gate that
// bit the set-rows-f16 smoke.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <vector>
#include <time.h>

extern "C" int ggml_cuda8_mul_mat_f16_f32_launch(
    const void  * src0, const float * src1, float * dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3);

// Host F32<->F16 so the CPU reference quantizes src0 identically to the kernel.
static uint16_t f32_to_fp16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    const uint32_t round_bit = (mant >> 12) & 1u;
    const uint32_t sticky    = (mant & 0xFFFu) != 0u;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    if (round_bit && (sticky || (h & 1u))) h += 1;
    return h;
}
static float fp16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { int e=-1; uint32_t m=mant; do{m<<=1;++e;}while((m&0x400)==0); m&=0x3FF;
               bits = sign | ((uint32_t)(127-15-e)<<23) | (m<<13); }
    } else if (exp == 0x1F) { bits = sign | 0x7F800000u | (mant << 13); }
    else { bits = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

struct Cfg {
    const char * label;
    int ne00, ne01, ne02, ne03;
    int ne11, ne12, ne13;
    bool timed;
};

static bool run_case(const Cfg & c) {
    const int ne00=c.ne00, ne01=c.ne01, ne02=c.ne02, ne03=c.ne03;
    const int ne11=c.ne11, ne12=c.ne12, ne13=c.ne13;
    const int r2 = ne12/ne02, r3 = ne13/ne03;

    // src0 F16 [ne00, ne01, ne02, ne03] packed; store as uint16 bits.
    const size_t n0 = (size_t) ne00*ne01*ne02*ne03;
    std::vector<uint16_t> h0(n0);
    std::vector<float>    h0f(n0);   // the F16-rounded float values, for the reference
    for (size_t i=0;i<n0;++i){
        const float v = ((float)((int)(i*2654435761u % 4001))/1000.0f) - 2.0f;
        h0[i]  = f32_to_fp16_bits(v);
        h0f[i] = fp16_bits_to_f32(h0[i]);   // what the kernel will actually see
    }
    // src1 F32 [ne00, ne11, ne12, ne13] packed
    const size_t n1 = (size_t) ne00*ne11*ne12*ne13;
    std::vector<float> h1(n1);
    for (size_t i=0;i<n1;++i) h1[i] = ((float)((int)(i*2971215073u % 4001))/1000.0f) - 2.0f;

    // dst F32 [ne01, ne11, ne12, ne13] packed
    const size_t nd = (size_t) ne01*ne11*ne12*ne13;

    // CPU reference (ggml dst layout: dst[i11..i13 row][i01]).
    std::vector<float> ref(nd, 0.0f);
    for (int i13=0;i13<ne13;++i13) for (int i12=0;i12<ne12;++i12)
    for (int i11=0;i11<ne11;++i11) for (int i01=0;i01<ne01;++i01) {
        const int i02 = i12/r2, i03 = i13/r3;
        const float * r1 = &h1[(((size_t)i13*ne12+i12)*ne11+i11)*ne00];
        const float * r0 = &h0f[(((size_t)i03*ne02+i02)*ne01+i01)*ne00];
        double s=0.0; for (int cc=0;cc<ne00;++cc) s += (double)r0[cc]*(double)r1[cc];
        ref[(((size_t)i13*ne12+i12)*ne11+i11)*ne01 + i01] = (float)s;
    }

    // Device tensors, packed strides.
    void * d0=NULL; float * d1=NULL; float * dd=NULL;
    cudaMalloc(&d0, n0*sizeof(uint16_t));
    cudaMalloc(&d1, n1*sizeof(float));
    cudaMalloc(&dd, nd*sizeof(float));
    cudaMemcpy(d0, &h0[0], n0*sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d1, &h1[0], n1*sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemset(dd, 0xFF, nd*sizeof(float));

    const size_t nb01=(size_t)ne00*2, nb02=nb01*ne01, nb03=nb02*ne02;          // F16: 2 bytes/elem
    const size_t nb11=(size_t)ne00*4, nb12=nb11*ne11, nb13=nb12*ne12;          // F32
    const size_t nb1 =(size_t)ne01*4, nb2 =nb1*ne11,  nb3 =nb2*ne12;           // F32 dst

    int rc = ggml_cuda8_mul_mat_f16_f32_launch(
        d0, d1, dd, ne00, ne01, ne02, ne03, ne11, ne12, ne13,
        nb01, nb02, nb03, nb11, nb12, nb13, nb1, nb2, nb3);

    bool ok = true;
    if (rc != 0) { std::printf("  %-40s FAIL (rc=%d)\n", c.label, rc); ok=false; }
    else {
        std::vector<float> got(nd);
        cudaMemcpy(&got[0], dd, nd*sizeof(float), cudaMemcpyDeviceToHost);
        double maxe=0; long bad=-1;
        for (size_t k=0;k<nd;++k){
            const double e=std::fabs((double)got[k]-(double)ref[k]);
            if (e>maxe) maxe=e;
            if (!approx(got[k], ref[k], 2e-3f) && bad<0) bad=(long)k;
        }
        double msg = 0.0;
        if (c.timed) {
            cudaDeviceSynchronize();
            struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
            const int iters=50;
            for (int it=0;it<iters;++it)
                ggml_cuda8_mul_mat_f16_f32_launch(d0,d1,dd,ne00,ne01,ne02,ne03,ne11,ne12,ne13,
                    nb01,nb02,nb03,nb11,nb12,nb13,nb1,nb2,nb3);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC,&t1);
            msg = ((t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6)/iters;
        }
        if (bad>=0){
            std::printf("  %-40s FAIL at %ld got=%.5g want=%.5g (maxe=%.3e)\n",
                        c.label, bad, (double)got[bad], (double)ref[bad], maxe);
            ok=false;
        } else if (c.timed){
            std::printf("  %-40s PASS (maxe=%.3e) | %.4f ms/call\n", c.label, maxe, msg);
        } else {
            std::printf("  %-40s PASS (maxe=%.3e)\n", c.label, maxe);
        }
    }
    cudaFree(d0); cudaFree(d1); cudaFree(dd);
    return ok;
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop,0)!=cudaSuccess){ std::fprintf(stderr,"no dev\n"); return 1; }
    std::printf("ggml-cuda8-mulmat-f16-smoke: %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    bool ok = true;

    std::printf("== F16 src0 x F32 src1 correctness ==\n");
    Cfg cases[] = {
        { "non-batched (ne02=ne12=1)",         64, 8,  1, 1,  4,  1, 1, false },
        { "batched no broadcast (4 heads)",     64, 8,  4, 1,  4,  4, 1, false },
        { "GQA broadcast (ne02=2, ne12=8)",     64, 16, 2, 1,  4,  8, 1, false },
        { "batch dim (ne03=2)",                 64, 8,  2, 2,  4,  2, 2, false },
        { "head_dim=128 (attn K.Q shape)",      128, 32, 4, 1, 8,  4, 1, false },
        { "n_kv=512 (attn probs.V shape)",      512, 8,  4, 1, 4,  4, 1, false },
    };
    for (size_t i=0;i<sizeof(cases)/sizeof(cases[0]);++i) ok &= run_case(cases[i]);

    std::printf("\n== perf (compare vs F32 mulmat 0.13 ms/call at similar shape) ==\n");
    Cfg timed[] = {
        { "head_dim=128, 32 rows, 8 heads",    128, 32, 8, 1, 8,  8, 1, true },
        { "n_kv=512, 8 rows, 4 heads",         512, 8,  4, 1, 4,  4, 1, true },
    };
    for (size_t i=0;i<sizeof(timed)/sizeof(timed[0]);++i) ok &= run_case(timed[i]);

    std::printf("\nggml-cuda8-mulmat-f16-smoke: %s\n",
                ok ? "SUCCESS (F16 read path correct)" : "FAIL");
    return ok ? 0 : 1;
}
