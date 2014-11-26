#include "cuda_helper.h"

typedef struct {
    uint32_t x[8][4];                     /*the 1024-bit state, ( x[i][0] || x[i][1] || x[i][2] || x[i][3] ) is the ith row of the state in the pseudocode*/
    uint32_t buffer[16];                  /*the 512-bit message block to be hashed;*/
} hashState;

/*42 round constants, each round constant is 32-byte (256-bit)*/
__constant__ uint32_t c_INIT_bitslice[8][4];
__constant__ unsigned char c_E8_bitslice_roundconstant[42][32];

const uint32_t h_INIT_bitslice[8][4] = {
	{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a},
	{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2},
	{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea},
	{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba},
	{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e},
	{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d},
	{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657},
	{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc} };

const unsigned char h_E8_bitslice_roundconstant[42][32]={
{0x72,0xd5,0xde,0xa2,0xdf,0x15,0xf8,0x67,0x7b,0x84,0x15,0xa,0xb7,0x23,0x15,0x57,0x81,0xab,0xd6,0x90,0x4d,0x5a,0x87,0xf6,0x4e,0x9f,0x4f,0xc5,0xc3,0xd1,0x2b,0x40},
{0xea,0x98,0x3a,0xe0,0x5c,0x45,0xfa,0x9c,0x3,0xc5,0xd2,0x99,0x66,0xb2,0x99,0x9a,0x66,0x2,0x96,0xb4,0xf2,0xbb,0x53,0x8a,0xb5,0x56,0x14,0x1a,0x88,0xdb,0xa2,0x31},
{0x3,0xa3,0x5a,0x5c,0x9a,0x19,0xe,0xdb,0x40,0x3f,0xb2,0xa,0x87,0xc1,0x44,0x10,0x1c,0x5,0x19,0x80,0x84,0x9e,0x95,0x1d,0x6f,0x33,0xeb,0xad,0x5e,0xe7,0xcd,0xdc},
{0x10,0xba,0x13,0x92,0x2,0xbf,0x6b,0x41,0xdc,0x78,0x65,0x15,0xf7,0xbb,0x27,0xd0,0xa,0x2c,0x81,0x39,0x37,0xaa,0x78,0x50,0x3f,0x1a,0xbf,0xd2,0x41,0x0,0x91,0xd3},
{0x42,0x2d,0x5a,0xd,0xf6,0xcc,0x7e,0x90,0xdd,0x62,0x9f,0x9c,0x92,0xc0,0x97,0xce,0x18,0x5c,0xa7,0xb,0xc7,0x2b,0x44,0xac,0xd1,0xdf,0x65,0xd6,0x63,0xc6,0xfc,0x23},
{0x97,0x6e,0x6c,0x3,0x9e,0xe0,0xb8,0x1a,0x21,0x5,0x45,0x7e,0x44,0x6c,0xec,0xa8,0xee,0xf1,0x3,0xbb,0x5d,0x8e,0x61,0xfa,0xfd,0x96,0x97,0xb2,0x94,0x83,0x81,0x97},
{0x4a,0x8e,0x85,0x37,0xdb,0x3,0x30,0x2f,0x2a,0x67,0x8d,0x2d,0xfb,0x9f,0x6a,0x95,0x8a,0xfe,0x73,0x81,0xf8,0xb8,0x69,0x6c,0x8a,0xc7,0x72,0x46,0xc0,0x7f,0x42,0x14},
{0xc5,0xf4,0x15,0x8f,0xbd,0xc7,0x5e,0xc4,0x75,0x44,0x6f,0xa7,0x8f,0x11,0xbb,0x80,0x52,0xde,0x75,0xb7,0xae,0xe4,0x88,0xbc,0x82,0xb8,0x0,0x1e,0x98,0xa6,0xa3,0xf4},
{0x8e,0xf4,0x8f,0x33,0xa9,0xa3,0x63,0x15,0xaa,0x5f,0x56,0x24,0xd5,0xb7,0xf9,0x89,0xb6,0xf1,0xed,0x20,0x7c,0x5a,0xe0,0xfd,0x36,0xca,0xe9,0x5a,0x6,0x42,0x2c,0x36},
{0xce,0x29,0x35,0x43,0x4e,0xfe,0x98,0x3d,0x53,0x3a,0xf9,0x74,0x73,0x9a,0x4b,0xa7,0xd0,0xf5,0x1f,0x59,0x6f,0x4e,0x81,0x86,0xe,0x9d,0xad,0x81,0xaf,0xd8,0x5a,0x9f},
{0xa7,0x5,0x6,0x67,0xee,0x34,0x62,0x6a,0x8b,0xb,0x28,0xbe,0x6e,0xb9,0x17,0x27,0x47,0x74,0x7,0x26,0xc6,0x80,0x10,0x3f,0xe0,0xa0,0x7e,0x6f,0xc6,0x7e,0x48,0x7b},
{0xd,0x55,0xa,0xa5,0x4a,0xf8,0xa4,0xc0,0x91,0xe3,0xe7,0x9f,0x97,0x8e,0xf1,0x9e,0x86,0x76,0x72,0x81,0x50,0x60,0x8d,0xd4,0x7e,0x9e,0x5a,0x41,0xf3,0xe5,0xb0,0x62},
{0xfc,0x9f,0x1f,0xec,0x40,0x54,0x20,0x7a,0xe3,0xe4,0x1a,0x0,0xce,0xf4,0xc9,0x84,0x4f,0xd7,0x94,0xf5,0x9d,0xfa,0x95,0xd8,0x55,0x2e,0x7e,0x11,0x24,0xc3,0x54,0xa5},
{0x5b,0xdf,0x72,0x28,0xbd,0xfe,0x6e,0x28,0x78,0xf5,0x7f,0xe2,0xf,0xa5,0xc4,0xb2,0x5,0x89,0x7c,0xef,0xee,0x49,0xd3,0x2e,0x44,0x7e,0x93,0x85,0xeb,0x28,0x59,0x7f},
{0x70,0x5f,0x69,0x37,0xb3,0x24,0x31,0x4a,0x5e,0x86,0x28,0xf1,0x1d,0xd6,0xe4,0x65,0xc7,0x1b,0x77,0x4,0x51,0xb9,0x20,0xe7,0x74,0xfe,0x43,0xe8,0x23,0xd4,0x87,0x8a},
{0x7d,0x29,0xe8,0xa3,0x92,0x76,0x94,0xf2,0xdd,0xcb,0x7a,0x9,0x9b,0x30,0xd9,0xc1,0x1d,0x1b,0x30,0xfb,0x5b,0xdc,0x1b,0xe0,0xda,0x24,0x49,0x4f,0xf2,0x9c,0x82,0xbf},
{0xa4,0xe7,0xba,0x31,0xb4,0x70,0xbf,0xff,0xd,0x32,0x44,0x5,0xde,0xf8,0xbc,0x48,0x3b,0xae,0xfc,0x32,0x53,0xbb,0xd3,0x39,0x45,0x9f,0xc3,0xc1,0xe0,0x29,0x8b,0xa0},
{0xe5,0xc9,0x5,0xfd,0xf7,0xae,0x9,0xf,0x94,0x70,0x34,0x12,0x42,0x90,0xf1,0x34,0xa2,0x71,0xb7,0x1,0xe3,0x44,0xed,0x95,0xe9,0x3b,0x8e,0x36,0x4f,0x2f,0x98,0x4a},
{0x88,0x40,0x1d,0x63,0xa0,0x6c,0xf6,0x15,0x47,0xc1,0x44,0x4b,0x87,0x52,0xaf,0xff,0x7e,0xbb,0x4a,0xf1,0xe2,0xa,0xc6,0x30,0x46,0x70,0xb6,0xc5,0xcc,0x6e,0x8c,0xe6},
{0xa4,0xd5,0xa4,0x56,0xbd,0x4f,0xca,0x0,0xda,0x9d,0x84,0x4b,0xc8,0x3e,0x18,0xae,0x73,0x57,0xce,0x45,0x30,0x64,0xd1,0xad,0xe8,0xa6,0xce,0x68,0x14,0x5c,0x25,0x67},
{0xa3,0xda,0x8c,0xf2,0xcb,0xe,0xe1,0x16,0x33,0xe9,0x6,0x58,0x9a,0x94,0x99,0x9a,0x1f,0x60,0xb2,0x20,0xc2,0x6f,0x84,0x7b,0xd1,0xce,0xac,0x7f,0xa0,0xd1,0x85,0x18},
{0x32,0x59,0x5b,0xa1,0x8d,0xdd,0x19,0xd3,0x50,0x9a,0x1c,0xc0,0xaa,0xa5,0xb4,0x46,0x9f,0x3d,0x63,0x67,0xe4,0x4,0x6b,0xba,0xf6,0xca,0x19,0xab,0xb,0x56,0xee,0x7e},
{0x1f,0xb1,0x79,0xea,0xa9,0x28,0x21,0x74,0xe9,0xbd,0xf7,0x35,0x3b,0x36,0x51,0xee,0x1d,0x57,0xac,0x5a,0x75,0x50,0xd3,0x76,0x3a,0x46,0xc2,0xfe,0xa3,0x7d,0x70,0x1},
{0xf7,0x35,0xc1,0xaf,0x98,0xa4,0xd8,0x42,0x78,0xed,0xec,0x20,0x9e,0x6b,0x67,0x79,0x41,0x83,0x63,0x15,0xea,0x3a,0xdb,0xa8,0xfa,0xc3,0x3b,0x4d,0x32,0x83,0x2c,0x83},
{0xa7,0x40,0x3b,0x1f,0x1c,0x27,0x47,0xf3,0x59,0x40,0xf0,0x34,0xb7,0x2d,0x76,0x9a,0xe7,0x3e,0x4e,0x6c,0xd2,0x21,0x4f,0xfd,0xb8,0xfd,0x8d,0x39,0xdc,0x57,0x59,0xef},
{0x8d,0x9b,0xc,0x49,0x2b,0x49,0xeb,0xda,0x5b,0xa2,0xd7,0x49,0x68,0xf3,0x70,0xd,0x7d,0x3b,0xae,0xd0,0x7a,0x8d,0x55,0x84,0xf5,0xa5,0xe9,0xf0,0xe4,0xf8,0x8e,0x65},
{0xa0,0xb8,0xa2,0xf4,0x36,0x10,0x3b,0x53,0xc,0xa8,0x7,0x9e,0x75,0x3e,0xec,0x5a,0x91,0x68,0x94,0x92,0x56,0xe8,0x88,0x4f,0x5b,0xb0,0x5c,0x55,0xf8,0xba,0xbc,0x4c},
{0xe3,0xbb,0x3b,0x99,0xf3,0x87,0x94,0x7b,0x75,0xda,0xf4,0xd6,0x72,0x6b,0x1c,0x5d,0x64,0xae,0xac,0x28,0xdc,0x34,0xb3,0x6d,0x6c,0x34,0xa5,0x50,0xb8,0x28,0xdb,0x71},
{0xf8,0x61,0xe2,0xf2,0x10,0x8d,0x51,0x2a,0xe3,0xdb,0x64,0x33,0x59,0xdd,0x75,0xfc,0x1c,0xac,0xbc,0xf1,0x43,0xce,0x3f,0xa2,0x67,0xbb,0xd1,0x3c,0x2,0xe8,0x43,0xb0},
{0x33,0xa,0x5b,0xca,0x88,0x29,0xa1,0x75,0x7f,0x34,0x19,0x4d,0xb4,0x16,0x53,0x5c,0x92,0x3b,0x94,0xc3,0xe,0x79,0x4d,0x1e,0x79,0x74,0x75,0xd7,0xb6,0xee,0xaf,0x3f},
{0xea,0xa8,0xd4,0xf7,0xbe,0x1a,0x39,0x21,0x5c,0xf4,0x7e,0x9,0x4c,0x23,0x27,0x51,0x26,0xa3,0x24,0x53,0xba,0x32,0x3c,0xd2,0x44,0xa3,0x17,0x4a,0x6d,0xa6,0xd5,0xad},
{0xb5,0x1d,0x3e,0xa6,0xaf,0xf2,0xc9,0x8,0x83,0x59,0x3d,0x98,0x91,0x6b,0x3c,0x56,0x4c,0xf8,0x7c,0xa1,0x72,0x86,0x60,0x4d,0x46,0xe2,0x3e,0xcc,0x8,0x6e,0xc7,0xf6},
{0x2f,0x98,0x33,0xb3,0xb1,0xbc,0x76,0x5e,0x2b,0xd6,0x66,0xa5,0xef,0xc4,0xe6,0x2a,0x6,0xf4,0xb6,0xe8,0xbe,0xc1,0xd4,0x36,0x74,0xee,0x82,0x15,0xbc,0xef,0x21,0x63},
{0xfd,0xc1,0x4e,0xd,0xf4,0x53,0xc9,0x69,0xa7,0x7d,0x5a,0xc4,0x6,0x58,0x58,0x26,0x7e,0xc1,0x14,0x16,0x6,0xe0,0xfa,0x16,0x7e,0x90,0xaf,0x3d,0x28,0x63,0x9d,0x3f},
{0xd2,0xc9,0xf2,0xe3,0x0,0x9b,0xd2,0xc,0x5f,0xaa,0xce,0x30,0xb7,0xd4,0xc,0x30,0x74,0x2a,0x51,0x16,0xf2,0xe0,0x32,0x98,0xd,0xeb,0x30,0xd8,0xe3,0xce,0xf8,0x9a},
{0x4b,0xc5,0x9e,0x7b,0xb5,0xf1,0x79,0x92,0xff,0x51,0xe6,0x6e,0x4,0x86,0x68,0xd3,0x9b,0x23,0x4d,0x57,0xe6,0x96,0x67,0x31,0xcc,0xe6,0xa6,0xf3,0x17,0xa,0x75,0x5},
{0xb1,0x76,0x81,0xd9,0x13,0x32,0x6c,0xce,0x3c,0x17,0x52,0x84,0xf8,0x5,0xa2,0x62,0xf4,0x2b,0xcb,0xb3,0x78,0x47,0x15,0x47,0xff,0x46,0x54,0x82,0x23,0x93,0x6a,0x48},
{0x38,0xdf,0x58,0x7,0x4e,0x5e,0x65,0x65,0xf2,0xfc,0x7c,0x89,0xfc,0x86,0x50,0x8e,0x31,0x70,0x2e,0x44,0xd0,0xb,0xca,0x86,0xf0,0x40,0x9,0xa2,0x30,0x78,0x47,0x4e},
{0x65,0xa0,0xee,0x39,0xd1,0xf7,0x38,0x83,0xf7,0x5e,0xe9,0x37,0xe4,0x2c,0x3a,0xbd,0x21,0x97,0xb2,0x26,0x1,0x13,0xf8,0x6f,0xa3,0x44,0xed,0xd1,0xef,0x9f,0xde,0xe7},
{0x8b,0xa0,0xdf,0x15,0x76,0x25,0x92,0xd9,0x3c,0x85,0xf7,0xf6,0x12,0xdc,0x42,0xbe,0xd8,0xa7,0xec,0x7c,0xab,0x27,0xb0,0x7e,0x53,0x8d,0x7d,0xda,0xaa,0x3e,0xa8,0xde},
{0xaa,0x25,0xce,0x93,0xbd,0x2,0x69,0xd8,0x5a,0xf6,0x43,0xfd,0x1a,0x73,0x8,0xf9,0xc0,0x5f,0xef,0xda,0x17,0x4a,0x19,0xa5,0x97,0x4d,0x66,0x33,0x4c,0xfd,0x21,0x6a},
{0x35,0xb4,0x98,0x31,0xdb,0x41,0x15,0x70,0xea,0x1e,0xf,0xbb,0xed,0xcd,0x54,0x9b,0x9a,0xd0,0x63,0xa1,0x51,0x97,0x40,0x72,0xf6,0x75,0x9d,0xbf,0x91,0x47,0x6f,0xe2}};

#define SWAP4(x,y)\
	y = (x & 0xf0f0f0f0UL); \
	x ^= y; \
	y >>= 4; \
	x <<= 4; \
	x |= y;

#define SWAP2(x,y)\
	y = (x & 0xccccccccUL); \
	x ^= y; \
	y >>= 2; \
	x <<= 2; \
	x |= y;

#define SWAP1(x,y)\
	y = (x & 0xaaaaaaaaUL); \
	x ^= y; \
	y >>= 1; \
	x += x; \
	x |= y;

/*swapping bits 16i||16i+1||......||16i+7  with bits 16i+8||16i+9||......||16i+15 of 32-bit x*/
//#define SWAP8(x)   (x) = ((((x) & 0x00ff00ffUL) << 8) | (((x) & 0xff00ff00UL) >> 8));
#define SWAP8(x) (x) = __byte_perm(x, x, 0x2301);
/*swapping bits 32i||32i+1||......||32i+15 with bits 32i+16||32i+17||......||32i+31 of 32-bit x*/
//#define SWAP16(x)  (x) = ((((x) & 0x0000ffffUL) << 16) | (((x) & 0xffff0000UL) >> 16));
#define SWAP16(x) (x) = __byte_perm(x, x, 0x1032);

/*The MDS transform*/
#define L(m0,m1,m2,m3,m4,m5,m6,m7) \
      (m4) ^= (m1);                \
      (m5) ^= (m2);                \
      (m6) ^= (m0) ^ (m3);         \
      (m7) ^= (m0);                \
      (m0) ^= (m5);                \
      (m1) ^= (m6);                \
      (m2) ^= (m4) ^ (m7);         \
      (m3) ^= (m4);

/*The Sbox*/
#define Sbox(m0,m1,m2,m3,cc)       \
      m3  = ~(m3);                 \
      m0 ^= ((~(m2)) & (cc));      \
      temp0 = (cc) ^ ((m0) & (m1));\
      m0 ^= ((m2) & (m3));         \
      m3 ^= ((~(m1)) & (m2));      \
      m1 ^= ((m0) & (m2));         \
      m2 ^= ((m0) & (~(m3)));      \
      m0 ^= ((m1) | (m3));         \
      m3 ^= ((m1) & (m2));         \
      m1 ^= (temp0 & (m0));        \
      m2 ^= temp0;

__device__ __forceinline__ void Sbox_and_MDS_layer(hashState* state, uint32_t roundnumber)
{
    uint32_t temp0;
	uint32_t cc0, cc1;
    //Sbox and MDS layer
#pragma unroll 4
    for (int i = 0; i < 4; i++) {
		cc0 = ((uint32_t*)c_E8_bitslice_roundconstant[roundnumber])[i];
		cc1 = ((uint32_t*)c_E8_bitslice_roundconstant[roundnumber])[i+4];
        Sbox(state->x[0][i],state->x[2][i], state->x[4][i], state->x[6][i], cc0);
        Sbox(state->x[1][i],state->x[3][i], state->x[5][i], state->x[7][i], cc1);
        L(state->x[0][i],state->x[2][i],state->x[4][i],state->x[6][i],state->x[1][i],state->x[3][i],state->x[5][i],state->x[7][i]);
    }
}

__device__ __forceinline__ void RoundFunction0(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		uint32_t y;
		SWAP1(state->x[j][0], y);
		SWAP1(state->x[j][1], y);
		SWAP1(state->x[j][2], y);
		SWAP1(state->x[j][3], y);
	}
}

__device__ __forceinline__ void RoundFunction1(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		uint32_t y;
		SWAP2(state->x[j][0], y);
		SWAP2(state->x[j][1], y);
		SWAP2(state->x[j][2], y);
		SWAP2(state->x[j][3], y);
	}
}

__device__ __forceinline__ void RoundFunction2(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		uint32_t y;
		SWAP4(state->x[j][0], y);
		SWAP4(state->x[j][1], y);
		SWAP4(state->x[j][2], y);
		SWAP4(state->x[j][3], y);
	}
}

__device__ __forceinline__ void RoundFunction3(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP8(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction4(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP16(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction5(hashState* state, uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 4; i = i+2) {
			temp0 = state->x[j][i]; state->x[j][i] = state->x[j][i+1]; state->x[j][i+1] = temp0;
		}
	}
}

__device__ __forceinline__ void RoundFunction6(hashState* state, uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 2; i++) {
			temp0 = state->x[j][i]; state->x[j][i] = state->x[j][i+2]; state->x[j][i+2] = temp0;
		}
	}
}

/*The bijective function E8, in bitslice form */
__device__ __forceinline__ void E8(hashState *state)
{
    /*perform 6 rounds*/
//#pragma unroll 6
    for (int i = 0; i < 42; i+=7)
	{
		RoundFunction0(state, i);
		RoundFunction1(state, i+1);
		RoundFunction2(state, i+2);
		RoundFunction3(state, i+3);
		RoundFunction4(state, i+4);
		RoundFunction5(state, i+5);
		RoundFunction6(state, i+6);
	}
}

/*The compression function F8 */
__device__ __forceinline__ void F8(hashState *state)
{
    /*xor the 512-bit message with the fist half of the 1024-bit hash state*/
#pragma unroll 16
    for (int i = 0; i < 16; i++)  state->x[i >> 2][i & 3] ^= ((uint32_t*)state->buffer)[i];

    /*the bijective function E8 */
    E8(state);

    /*xor the 512-bit message with the second half of the 1024-bit hash state*/
#pragma unroll 16
    for (int i = 0; i < 16; i++)  state->x[(16+i) >> 2][(16+i) & 3] ^= ((uint32_t*)state->buffer)[i];
}


__device__ __forceinline__ void JHHash(const uint32_t *data, uint32_t *hashval)
{
    hashState state;

    /*load the intital hash value H0 into state*/
	/*
    #define INIT(a,b,c,d) ((a) | ((b)<<8) | ((c)<<16) | ((d)<<24))
    state.x[0][0] = INIT(0x6f,0xd1,0x4b,0x96);
    state.x[0][1] = INIT(0x3e,0x00,0xaa,0x17);
    state.x[0][2] = INIT(0x63,0x6a,0x2e,0x05);
    state.x[0][3] = INIT(0x7a,0x15,0xd5,0x43);
    state.x[1][0] = INIT(0x8a,0x22,0x5e,0x8d);
    state.x[1][1] = INIT(0x0c,0x97,0xef,0x0b);
    state.x[1][2] = INIT(0xe9,0x34,0x12,0x59);
    state.x[1][3] = INIT(0xf2,0xb3,0xc3,0x61);
    state.x[2][0] = INIT(0x89,0x1d,0xa0,0xc1);
    state.x[2][1] = INIT(0x53,0x6f,0x80,0x1e);
    state.x[2][2] = INIT(0x2a,0xa9,0x05,0x6b);
    state.x[2][3] = INIT(0xea,0x2b,0x6d,0x80);
    state.x[3][0] = INIT(0x58,0x8e,0xcc,0xdb);
    state.x[3][1] = INIT(0x20,0x75,0xba,0xa6);
    state.x[3][2] = INIT(0xa9,0x0f,0x3a,0x76);
    state.x[3][3] = INIT(0xba,0xf8,0x3b,0xf7);
    state.x[4][0] = INIT(0x01,0x69,0xe6,0x05);
    state.x[4][1] = INIT(0x41,0xe3,0x4a,0x69);
    state.x[4][2] = INIT(0x46,0xb5,0x8a,0x8e);
    state.x[4][3] = INIT(0x2e,0x6f,0xe6,0x5a);
    state.x[5][0] = INIT(0x10,0x47,0xa7,0xd0);
    state.x[5][1] = INIT(0xc1,0x84,0x3c,0x24);
    state.x[5][2] = INIT(0x3b,0x6e,0x71,0xb1);
    state.x[5][3] = INIT(0x2d,0x5a,0xc1,0x99);
    state.x[6][0] = INIT(0xcf,0x57,0xf6,0xec);
    state.x[6][1] = INIT(0x9d,0xb1,0xf8,0x56);
    state.x[6][2] = INIT(0xa7,0x06,0x88,0x7c);
    state.x[6][3] = INIT(0x57,0x16,0xb1,0x56);
    state.x[7][0] = INIT(0xe3,0xc2,0xfc,0xdf);
    state.x[7][1] = INIT(0xe6,0x85,0x17,0xfb);
    state.x[7][2] = INIT(0x54,0x5a,0x46,0x78);
    state.x[7][3] = INIT(0xcc,0x8c,0xdd,0x4b);
	*/
#pragma unroll 8
	for(int j=0;j<8;j++)
	{
#pragma unroll 4
		for(int i=0;i<4;i++)
			state.x[j][i] = c_INIT_bitslice[j][i];
	}

#pragma unroll 16
    for (int i=0; i < 16; ++i) state.buffer[i] = data[i];
    F8(&state);

    /*pad the message when databitlen is multiple of 512 bits, then process the padded block*/
    state.buffer[0] = 0x80;
#pragma unroll 14
    for (int i=1; i < 15; i++) state.buffer[i] = 0;
    state.buffer[15] = 0x00020000;
    F8(&state);

    /*truncating the final hash value to generate the message digest*/
#pragma unroll 16
    for (int i=0; i < 16; ++i) hashval[i] = state.x[4][i];
}

// Die Hash-Funktion
__global__ __launch_bounds__(256, 3)
void quark_jh512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

        JHHash(Hash, Hash);
    }
}


// Setup-Funktionen
__host__ void quark_jh512_cpu_init(int thr_id, int threads)
{
	
    cudaMemcpyToSymbol( c_E8_bitslice_roundconstant,
                        h_E8_bitslice_roundconstant,
                        sizeof(h_E8_bitslice_roundconstant),
                        0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( c_INIT_bitslice,
                        h_INIT_bitslice,
                        sizeof(h_INIT_bitslice),
                        0, cudaMemcpyHostToDevice);
}

__host__ void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Gr��e des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_jh512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
    MyStreamSynchronize(NULL, order, thr_id);
}

