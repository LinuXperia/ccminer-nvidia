/* XMRig
* Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
* Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
* Copyright 2014      Lucas Jones <https://github.com/lucasjones>
* Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
* Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
* Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
* Copyright 2018      Lee Clagett <https://github.com/vtnerd>
* Copyright 2016-2018 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include <stdint.h>


#include "xmrig.h"


typedef struct {
    int device_id;
    const char *device_name;
    int device_arch[2];
    int device_mpcount;
    int device_blocks;
    int device_threads;
    int device_bfactor;
    int device_bsleep;
    int device_clockRate;
    int device_memoryClockRate;
    uint32_t device_pciBusID;
    uint32_t device_pciDeviceID;
    uint32_t device_pciDomainID;
    uint32_t syncMode;

    uint32_t *d_input;
    uint32_t inputlen;
    uint32_t *d_result_count;
    uint32_t *d_result_nonce;
    uint32_t *d_long_state;
    uint32_t *d_ctx_state;
    uint32_t *d_ctx_state2;
    uint32_t *d_ctx_a;
    uint32_t *d_ctx_b;
    uint32_t *d_ctx_key1;
    uint32_t *d_ctx_key2;
    uint32_t *d_ctx_text;
    uint32_t *d_tweak1_2;
} nvid_ctx;


int cuda_get_devicecount();
int cuda_get_runtime_version();
int cuda_get_deviceinfo(nvid_ctx *ctx, xmrig::Algo algo, bool isCNv2);
int cryptonight_gpu_init(nvid_ctx *ctx, xmrig::Algo algo);
void cryptonight_extra_cpu_set_data(nvid_ctx *ctx, const void *data, size_t len);
void cryptonight_extra_cpu_prepare(nvid_ctx *ctx, uint32_t startNonce, xmrig::Algo algo, xmrig::Variant variant);
void cryptonight_gpu_hash(nvid_ctx *ctx, xmrig::Algo algo, xmrig::Variant variant, uint32_t startNonce);
void cryptonight_extra_cpu_final(nvid_ctx *ctx, uint32_t startNonce, uint64_t target, uint32_t *rescount, uint32_t *resnonce, xmrig::Algo algo);
void cryptonight_extra_cpu_free(nvid_ctx *ctx, xmrig::Algo algo);


#include <cuda_runtime.h>
#include <miner.h>

#define MEMORY         (1U << 21) // 2 MiB / 2097152 B
#define ITER           (1U << 20) // 1048576
#define E2I_MASK       0x1FFFF0u

#define AES_BLOCK_SIZE  16U
#define AES_KEY_SIZE    32
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE) // 128 B

#define AES_RKEY_LEN 4
#define AES_COL_LEN 4
#define AES_ROUND_BASE 7

#ifndef HASH_SIZE
#define HASH_SIZE 32
#endif

#ifndef HASH_DATA_AREA
#define HASH_DATA_AREA 136
#endif

#define hi_dword(x) (x >> 32)
#define lo_dword(x) (x & 0xFFFFFFFF)

#define C32(x)    ((uint32_t)(x ## U))
#define T32(x) ((x) & C32(0xFFFFFFFF))

#ifndef ROTL64
    #if __CUDA_ARCH__ >= 350
        __forceinline__ __device__ uint64_t cuda_ROTL64(const uint64_t value, const int offset) {
            uint2 result;
            if(offset >= 32) {
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
            } else {
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
            }
            return  __double_as_longlong(__hiloint2double(result.y, result.x));
        }
        #define ROTL64(x, n) (cuda_ROTL64(x, n))
    #else
        #define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
    #endif
#endif

#ifndef ROTL32
    #if __CUDA_ARCH__ < 350
        #define ROTL32(x, n) T32(((x) << (n)) | ((x) >> (32 - (n))))
    #else
        #define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
    #endif
#endif

#ifndef ROTR32
    #if __CUDA_ARCH__ < 350
        #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #else
        #define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
    #endif
#endif

#define MEMSET8(dst,what,cnt) { \
    int i_memset8; \
    uint64_t *out_memset8 = (uint64_t *)(dst); \
    for( i_memset8 = 0; i_memset8 < cnt; i_memset8++ ) \
        out_memset8[i_memset8] = (what); }

#define MEMSET4(dst,what,cnt) { \
    int i_memset4; \
    uint32_t *out_memset4 = (uint32_t *)(dst); \
    for( i_memset4 = 0; i_memset4 < cnt; i_memset4++ ) \
        out_memset4[i_memset4] = (what); }

#define MEMCPY8(dst,src,cnt) { \
    int i_memcpy8; \
    uint64_t *in_memcpy8 = (uint64_t *)(src); \
    uint64_t *out_memcpy8 = (uint64_t *)(dst); \
    for( i_memcpy8 = 0; i_memcpy8 < cnt; i_memcpy8++ ) \
        out_memcpy8[i_memcpy8] = in_memcpy8[i_memcpy8]; }

#define MEMCPY4(dst,src,cnt) { \
    int i_memcpy4; \
    uint32_t *in_memcpy4 = (uint32_t *)(src); \
    uint32_t *out_memcpy4 = (uint32_t *)(dst); \
    for( i_memcpy4 = 0; i_memcpy4 < cnt; i_memcpy4++ ) \
        out_memcpy4[i_memcpy4] = in_memcpy4[i_memcpy4]; }

#define XOR_BLOCKS_DST(x,y,z) { \
    ((uint64_t *)z)[0] = ((uint64_t *)(x))[0] ^ ((uint64_t *)(y))[0]; \
    ((uint64_t *)z)[1] = ((uint64_t *)(x))[1] ^ ((uint64_t *)(y))[1]; }

#define E2I(x) ((size_t)(((*((uint64_t*)(x)) >> 4) & 0x1ffff)))

union hash_state {
  uint8_t b[200];
  uint64_t w[25];
};

union cn_slow_hash_state {
    union hash_state hs;
    struct {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};

static inline void exit_if_cudaerror(int thr_id, const char *src, int line)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "%s %s line %d", cudaGetErrorString(err), src, line);
		exit(1);
	}
}
