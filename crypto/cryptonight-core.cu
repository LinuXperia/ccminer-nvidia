#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#undef __shfl
#define __shfl(var, srcLane, width) __shfl_sync(0xFFFFFFFFu, var, srcLane, width)
#endif

#include "cryptonight.h"

#define LONG_SHL32 19 // 1<<19 (uint32_t* index)
#define LONG_SHL64 18 // 1<<18 (uint64_t* index)
#define LONG_LOOPS32 0x80000U

#include "cn_aes.cuh"

typedef int IndexType;

__global__
void cryptonight_gpu_phase1(const uint32_t threads, uint32_t * __restrict__ d_long_state,
	uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1)
{
	__shared__ uint32_t sharedMemory[1024];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	if(thread < threads)
	{
		cn_aes_gpu_init(sharedMemory);
		__syncthreads();

		const uint32_t sub = (threadIdx.x & 0x7U) << 2;
		uint32_t *longstate = &d_long_state[(thread << LONG_SHL32) + sub];
		uint32_t __align__(8) key[40];
		MEMCPY8(key, &ctx_key1[thread * 40U], 20);
		uint32_t __align__(8) text[4];
		MEMCPY8(text, &ctx_state[thread * 50U + sub + 16U], 2);

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			MEMCPY8(&longstate[i], text, 2);
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ ulonglong2 cuda_mul128(const uint64_t multiplier, const uint64_t multiplicand)
{
	ulonglong2 product;
	product.x = __umul64hi(multiplier, multiplicand);
	product.y = multiplier * multiplicand;
	return product;
}

static __forceinline__ __device__ void operator += (ulonglong2 &a, const ulonglong2 b) {
	a.x += b.x; a.y += b.y;
}

static __forceinline__ __device__ ulonglong2 operator ^ (const ulonglong2 &a, const ulonglong2 &b) {
	return make_ulonglong2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ void MUL_SUM_XOR_DST_0(const uint64_t m, uint4 &a, void* far_dst)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	AS_UL2(far_dst) = p;
}

__global__
#if __CUDA_ARCH__ >= 500
//__launch_bounds__(128,12) /* force 40 regs to allow -l ...x32 */
#endif
void cryptonight_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;

		void * ctx_a = (void*)(&d_ctx_a[thread << 2U]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2U]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;

			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			AS_UINT4(&long_state[j]) = C ^ B; // st.global.u32.v4
			MUL_SUM_XOR_DST_0((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3]);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			AS_UINT4(&long_state[j]) = C ^ B;
			MUL_SUM_XOR_DST_0((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3]);
		}

		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void store_variant1(uint64_t* long_state, uint4 Z)
{
	const uint32_t tmp = (Z.z >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 3) & 6u) | (tmp & 1u)) << 1;
	Z.z = (Z.z & 0x00ffffffu) | ((tmp ^ ((0x75310u >> index) & 0x30u)) << 24);
	AS_UINT4(long_state) = Z;
}

__device__ __forceinline__ void store_variant2(uint64_t* long_state, uint4 Z)
{
	const uint32_t tmp = (Z.z >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 4) & 6u) | (tmp & 1u)) << 1;
	Z.z = (Z.z & 0x00ffffffu) | ((tmp ^ ((0x75312u >> index) & 0x30u)) << 24);
	AS_UINT4(long_state) = Z;
}

__device__ __forceinline__ void MUL_SUM_XOR_DST_1(const uint64_t m, uint4 &a, void* far_dst, uint64_t tweak)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	p.y = p.y ^ tweak;
	AS_UL2(far_dst) = p;
}

constexpr const uint32_t CRYPTONIGHT_MASK         = 0x1FFFF0;
constexpr const uint32_t CRYPTONIGHT_ITER         = 0x80000;
//constexpr const size_t   CRYPTONIGHT_MEMORY       = 2 * 1024 * 1024;
constexpr const size_t   CRYPTONIGHT_MEMORY       = 1024 * 1024;

struct u64 : public uint2
{

    __forceinline__ __device__ u64(){}

    __forceinline__ __device__ u64( const uint32_t x0, const uint32_t x1)
    {
        uint2::x = x0;
        uint2::y = x1;
    }

    __forceinline__ __device__ operator uint64_t() const
    {
        return *((uint64_t*)this);
    }

    __forceinline__ __device__ u64( const uint64_t x0)
    {
        ((uint64_t*)&this->x)[0] = x0;
    }

    __forceinline__ __device__ u64 operator^=(const u64& other)
    {
        uint2::x ^= other.x;
        uint2::y ^= other.y;

        return *this;
    }

    __forceinline__ __device__ u64 operator+(const u64& other) const
    {
        u64 tmp;
        ((uint64_t*)&tmp.x)[0] = ((uint64_t*)&(this->x))[0] + ((uint64_t*)&(other.x))[0];

        return tmp;
    }

    __forceinline__ __device__ u64 operator+=(const uint64_t& other)
    {
        return ((uint64_t*)&this->x)[0] += other;
    }

    __forceinline__ __device__ void print(int i) const
    {
        if(i<2)
            printf("gpu: %lu\n", ((uint64_t*)&this->x)[0]);
    }
};

/** avoid warning `unused parameter` */
template< typename T >
__forceinline__ __device__ void unusedVar( const T& )
{
}

__device__ __forceinline__ uint32_t get_reciprocal(uint32_t a)
{
	const float a_hi = __uint_as_float((a >> 8) + ((126U + 31U) << 23));
	const float a_lo = __uint2float_rn(a & 0xFF);

	float r;
	asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(a_hi));
	const float r_scaled = __uint_as_float(__float_as_uint(r) + (64U << 23));

	const float h = __fmaf_rn(a_lo, r, __fmaf_rn(a_hi, r, -1.0f));
	return (__float_as_uint(r) << 9) - __float2int_rn(h * r_scaled);
}

/** shuffle data for
 *
 * - this method can be used with all compute architectures
 * - for <sm_30 shared memory is needed
 *
 * group_n - must be a power of 2!
 *
 * @param ptr pointer to shared memory, size must be `threadIdx.x * sizeof(uint32_t)`
 *            value can be NULL for compute architecture >=sm_30
 * @param sub thread number within the group, range [0:group_n]
 * @param value value to share with other threads within the group
 * @param src thread number within the group from where the data is read, range [0:group_n]
 */
template<size_t group_n>
__forceinline__ __device__ uint32_t shuffle(volatile uint32_t* ptr,const uint32_t sub,const int val,const uint32_t src)
{
#   if ( __CUDA_ARCH__ < 300 )
    ptr[sub] = val;
    return ptr[src & (group_n-1)];
#   else
    unusedVar( ptr );
    unusedVar( sub );
#   if (__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_sync(__activemask(), val, src, group_n);
#   else
    return __shfl( val, src, group_n );
#   endif
#   endif
}

__device__ __forceinline__ uint64_t fast_div_v2(uint64_t a, uint32_t b)
{
	const uint32_t r = get_reciprocal(b);
	const uint64_t k = __umulhi(((uint32_t*)&a)[0], r) + ((uint64_t)(r) * ((uint32_t*)&a)[1]) + a;

	uint32_t q[2];
	q[0] = ((uint32_t*)&k)[1];

	int64_t tmp = a - (uint64_t)(q[0]) * b;
	((int32_t*)(&tmp))[1] -= (k < a) ? b : 0;

	const bool overshoot = ((int32_t*)(&tmp))[1] < 0;
	const bool undershoot = tmp >= b;

	q[0] += (undershoot ? 1U : 0U) - (overshoot ? 1U : 0U);
	q[1] = ((uint32_t*)(&tmp))[0] + (overshoot ? b : 0U) - (undershoot ? b : 0U);

	return *((uint64_t*)(q));
}

__device__ __forceinline__ uint32_t fast_sqrt_v2(const uint64_t n1)
{
	float x = __uint_as_float((((uint32_t*)&n1)[1] >> 9) + ((64U + 127U) << 23));
	float x1;
	asm("rsqrt.approx.f32 %0, %1;" : "=f"(x1) : "f"(x));
	asm("sqrt.approx.f32 %0, %1;" : "=f"(x) : "f"(x));

	// The following line does x1 *= 4294967296.0f;
	x1 = __uint_as_float(__float_as_uint(x1) + (32U << 23));

	const uint32_t x0 = __float_as_uint(x) - (158U << 23);
	const int64_t delta0 = n1 - (((int64_t)(x0) * x0) << 18);
	const float delta = __int2float_rn(((int32_t*)&delta0)[1]) * x1;

	uint32_t result = (x0 << 10) + __float2int_rn(delta);
	const uint32_t s = result >> 1;
	const uint32_t b = result & 1;

	const uint64_t x2 = (uint64_t)(s) * (s + b) + ((uint64_t)(result) << 32) - n1;
	if ((int64_t)(x2 + b) > 0) --result;
	if ((int64_t)(x2 + 0x100000000UL + s) < 0) ++result;

	return result;
}

constexpr size_t MASK       = CRYPTONIGHT_MASK;
constexpr size_t ITERATIONS = CRYPTONIGHT_ITER;
constexpr size_t MEM        = CRYPTONIGHT_MEMORY;

__global__
void monero_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );

#   if( __CUDA_ARCH__ < 300 )
	extern __shared__ uint64_t externShared[];
	// 8 x 64bit values
	volatile uint64_t* myChunks = (volatile uint64_t*)(externShared + (threadIdx.x >> 1) * 8);
	volatile uint32_t* sPtr = (volatile uint32_t*)(externShared + (blockDim.x >> 1) * 8)  + (threadIdx.x & 0xFFFFFFFE);
#   else
	extern __shared__ uint64_t chunkMem[];
	volatile uint32_t* sPtr = NULL;
	// 8 x 64bit values
	volatile uint64_t* myChunks = (volatile uint64_t*)(chunkMem + (threadIdx.x >> 1) * 8);
#   endif

	__syncthreads( );

	const uint64_t tid    = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t thread = tid >> 1;
	const uint32_t sub    = tid & 1;

	if (thread >= threads) {
			return;
	}

	uint8_t *l0              = (uint8_t*)&d_long_state[(IndexType) thread * MEM];
	uint64_t ax0             = ((uint64_t*)(d_ctx_a + thread * 4))[sub];
	uint32_t idx0            = shuffle<2>(sPtr, sub, static_cast<uint32_t>(ax0), 0);
	uint64_t bx0             = ((uint64_t*)(d_ctx_b + thread * 12))[sub];
	uint64_t bx1             = ((uint64_t*)(d_ctx_b + thread * 12 + 4))[sub];
	uint64_t division_result = ((uint64_t*)(d_ctx_b + thread * 12 + 4 * 2))[0];
	uint32_t sqrt_result     = (d_ctx_b + thread * 12 + 4 * 2 + 2)[0];

	const int batchsize      = (ITERATIONS * 2) >> ( 1 + bfactor );
	const int start          = partidx * batchsize;
	const int end            = start + batchsize;

	uint64_t* ptr0;
	for (int i = start; i < end; ++i) {
			ptr0 = (uint64_t *)&l0[idx0 & 0x1FFFC0];

			((ulonglong4*)myChunks)[sub] = ((ulonglong4*)ptr0)[sub];

			uint32_t idx1 = (idx0 & 0x30) >> 3;
			const u64 cx  = myChunks[ idx1 + sub ];
			const u64 cx2 = myChunks[ idx1 + ((sub + 1) & 1) ];

			u64 cx_aes = ax0 ^ u64(
					t_fn0( cx.x & 0xff ) ^ t_fn1( (cx.y >> 8) & 0xff ) ^ t_fn2( (cx2.x >> 16) & 0xff ) ^ t_fn3( (cx2.y >> 24 ) ),
					t_fn0( cx.y & 0xff ) ^ t_fn1( (cx2.x >> 8) & 0xff ) ^ t_fn2( (cx2.y >> 16) & 0xff ) ^ t_fn3( (cx.x >> 24 ) )
			);

			{
					const uint64_t chunk1 = myChunks[idx1 ^ 2 + sub];
					const uint64_t chunk2 = myChunks[idx1 ^ 4 + sub];
					const uint64_t chunk3 = myChunks[idx1 ^ 6 + sub];

#           if (__CUDACC_VER_MAJOR__ >= 9)
					__syncwarp();
#           else
					__syncthreads();
#           endif

					myChunks[idx1 ^ 2 + sub] = chunk3 + bx1;
					myChunks[idx1 ^ 4 + sub] = chunk1 + bx0;
					myChunks[idx1 ^ 6 + sub] = chunk2 + ax0;
			}

			myChunks[idx1 + sub] = cx_aes ^ bx0;

			((ulonglong4*)ptr0)[sub] = ((ulonglong4*)myChunks)[sub];

			idx0 = shuffle<2>(sPtr, sub, cx_aes.x, 0);
			idx1 = (idx0 & 0x30) >> 3;
			ptr0 = (uint64_t *)&l0[idx0 & MASK & 0x1FFFC0];

			((ulonglong4*)myChunks)[sub] = ((ulonglong4*)ptr0)[sub];

			uint64_t cx_mul;
			((uint32_t*)&cx_mul)[0] = shuffle<2>(sPtr, sub, cx_aes.x , 0);
			((uint32_t*)&cx_mul)[1] = shuffle<2>(sPtr, sub, cx_aes.y , 0);

			if (sub == 1) {
					// Use division and square root results from the _previous_ iteration to hide the latency
					((uint32_t*)&division_result)[1] ^= sqrt_result;
					((uint64_t*)myChunks)[idx1]      ^= division_result;

					const uint32_t dd = (static_cast<uint32_t>(cx_mul) + (sqrt_result << 1)) | 0x80000001UL;
					division_result = fast_div_v2(cx_aes, dd);

					// Use division_result as an input for the square root to prevent parallel implementation in hardware
					sqrt_result = fast_sqrt_v2(cx_mul + division_result);
			}

#       if (__CUDACC_VER_MAJOR__ >= 9)
			__syncwarp();
#       else
			__syncthreads( );
#       endif

			uint64_t c = ((uint64_t*)myChunks)[idx1 + sub];

			{
					uint64_t cl = ((uint64_t*)myChunks)[idx1];
					// sub 0 -> hi, sub 1 -> lo
					uint64_t res = sub == 0 ? __umul64hi( cx_mul, cl ) : cx_mul * cl;

					const uint64_t chunk1 = myChunks[ idx1 ^ 2 + sub ] ^ res;
					uint64_t chunk2       = myChunks[ idx1 ^ 4 + sub ];
					res ^= ((uint64_t*)&chunk2)[0];
					const uint64_t chunk3 = myChunks[ idx1 ^ 6 + sub ];

#           if (__CUDACC_VER_MAJOR__ >= 9)
					__syncwarp();
#           else
					__syncthreads( );
#           endif

					myChunks[idx1 ^ 2 + sub] = chunk3 + bx1;
					myChunks[idx1 ^ 4 + sub] = chunk1 + bx0;
					myChunks[idx1 ^ 6 + sub] = chunk2 + ax0;

					ax0 += res;
			}

			bx1 = bx0;
			bx0 = cx_aes;

			myChunks[idx1 + sub] = ax0;

			((ulonglong4*)ptr0)[sub] = ((ulonglong4*)myChunks)[sub];

			ax0 ^= c;
			idx0 = shuffle<2>(sPtr, sub, static_cast<uint32_t>(ax0), 0);
	}

	if (bfactor > 0) {
			((uint64_t*)(d_ctx_a + thread * 4))[sub]      = ax0;
			((uint64_t*)(d_ctx_b + thread * 12))[sub]     = bx0;
			((uint64_t*)(d_ctx_b + thread * 12 + 4))[sub] = bx1;

			if (sub == 1) {
					// must be valid only for `sub == 1`
					((uint64_t*)(d_ctx_b + thread * 12 + 4 * 2))[0] = division_result;
					(d_ctx_b + thread * 12 + 4 * 2 + 2)[0]          = sqrt_result;
			}
	}
}

// --------------------------------------------------------------------------------------------------------------

__global__
void stellite_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;
		uint64_t tweak = d_tweak[thread];

		void * ctx_a = (void*)(&d_ctx_a[thread << 2]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;
			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			store_variant2(&long_state[j], C ^ B); // st.global
			MUL_SUM_XOR_DST_1((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3], tweak);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			store_variant2(&long_state[j], C ^ B);
			MUL_SUM_XOR_DST_1((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3], tweak);
		}
		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__global__
void cryptonight_gpu_phase3(const uint32_t threads, const uint32_t * __restrict__ d_long_state,
	uint32_t * __restrict__ d_ctx_state, const uint32_t * __restrict__ d_ctx_key2)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;

	if(thread < threads)
	{
		const int sub = (threadIdx.x & 7) << 2;
		const uint32_t *longstate = &d_long_state[(thread << LONG_SHL32) + sub];
		uint32_t key[40], text[4];
		MEMCPY8(key, d_ctx_key2 + thread * 40, 20);
		MEMCPY8(text, d_ctx_state + thread * 50 + sub + 16, 2);

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			#pragma unroll
			for(int j = 0; j < 4; ++j)
				text[j] ^= longstate[i + j];

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		MEMCPY8(d_ctx_state + thread * 50 + sub + 16, text, 2);
	}
}

// --------------------------------------------------------------------------------------------------------------

extern int device_bfactor[MAX_GPUS];

__host__
void cryptonight_core_cuda(int thr_id, uint32_t blocks, uint32_t threads, uint64_t *d_long_state, uint32_t *d_ctx_state,
	uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint64_t *d_ctx_tweak)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block2(threads << 2);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	const uint16_t bfactor = (uint16_t) device_bfactor[thr_id];
	const uint32_t partcount = 1U << bfactor;
	const uint32_t throughput = (uint32_t) (blocks*threads);

	const int bsleep = bfactor ? 100 : 0;
	const int dev_id = device_map[thr_id];

	cryptonight_gpu_phase1 <<<grid, block8>>> (throughput, (uint32_t*) d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	if(partcount > 1) usleep(bsleep);

	for (uint32_t i = 0; i < partcount; i++)
	{
		dim3 b = device_sm[dev_id] >= 300 ? block4 : block;
		if (variant == 0) {
			cryptonight_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b);
		}	else if (variant == 2 || cryptonight_fork == 8) {
			monero_gpu_phase2 <<<
				grid,
				block2,
				sizeof(uint64_t) * block2.x * 8 + block2.x * sizeof(uint32_t) * static_cast<int>(device_sm[dev_id] < 300)
			>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
		} else if (variant == 2 && cryptonight_fork == 3) {
			stellite_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
		}
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		if(partcount > 1) usleep(bsleep);
	}
	//cudaDeviceSynchronize();
	//exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	cryptonight_gpu_phase3 <<<grid, block8>>> (throughput, (uint32_t*) d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
}
