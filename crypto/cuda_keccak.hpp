#define KECCAK_ROUNDS	24

#if __CUDA_ARCH__ >= 350
	__forceinline__ __device__ uint64_t cuda_rotl64(const uint64_t value, const int offset)
	{
		uint2 result;
		if(offset >= 32)
		{
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		}
		else
		{
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
			asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		}
		return  __double_as_longlong(__hiloint2double(result.y, result.x));
	}
	#define rotl64_1(x, y) (cuda_rotl64((x), (y)))
#else
	#define rotl64_1(x, y) ((x) << (y) | ((x) >> (64 - (y))))
#endif

#define rotl64_2(x, y) rotl64_1(((x) >> 32) | ((x) << 32), (y))
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__device__ __forceinline__
void cn_keccakf2(uint64_t *s)
{
	uint8_t i;

	for(i = 0; i < 24; ++i)
	{
		uint64_t bc[5], tmpxor[5], tmp1, tmp2;

		tmpxor[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		tmpxor[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		tmpxor[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		tmpxor[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		tmpxor[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		bc[0] = tmpxor[0] ^ rotl64_1(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ rotl64_1(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ rotl64_1(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ rotl64_1(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ rotl64_1(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = rotl64_2(s[6] ^ bc[0], 12);
		s[6] = rotl64_1(s[9] ^ bc[3], 20);
		s[9] = rotl64_2(s[22] ^ bc[1], 29);
		s[22] = rotl64_2(s[14] ^ bc[3], 7);
		s[14] = rotl64_1(s[20] ^ bc[4], 18);
		s[20] = rotl64_2(s[2] ^ bc[1], 30);
		s[2] = rotl64_2(s[12] ^ bc[1], 11);
		s[12] = rotl64_1(s[13] ^ bc[2], 25);
		s[13] = rotl64_1(s[19] ^ bc[3], 8);
		s[19] = rotl64_2(s[23] ^ bc[2], 24);
		s[23] = rotl64_2(s[15] ^ bc[4], 9);
		s[15] = rotl64_1(s[4] ^ bc[3], 27);
		s[4] = rotl64_1(s[24] ^ bc[3], 14);
		s[24] = rotl64_1(s[21] ^ bc[0], 2);
		s[21] = rotl64_2(s[8] ^ bc[2], 23);
		s[8] = rotl64_2(s[16] ^ bc[0], 13);
		s[16] = rotl64_2(s[5] ^ bc[4], 4);
		s[5] = rotl64_1(s[3] ^ bc[2], 28);
		s[3] = rotl64_1(s[18] ^ bc[2], 21);
		s[18] = rotl64_1(s[17] ^ bc[1], 15);
		s[17] = rotl64_1(s[11] ^ bc[0], 10);
		s[11] = rotl64_1(s[7] ^ bc[1], 6);
		s[7] = rotl64_1(s[10] ^ bc[4], 3);
		s[10] = rotl64_1(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0] ^= keccakf_rndc[i];
	}
}

__constant__ int keccakf_rotc[24] =
{
		1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
		27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int keccakf_piln[24] =
{
		10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
		15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

__device__ __forceinline__
void cn_keccakf(uint64_t st[25], int rounds)
{
    int i, j, round;
    uint64_t t, bc[5];

    for (round = 0; round < rounds; ++round) {

        // Theta
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        for (i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            st[i     ] ^= t;
            st[i +  5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        // Rho Pi
        t = st[1];
        for (i = 0; i < 24; ++i) {
            bc[0] = st[keccakf_piln[i]];
            st[keccakf_piln[i]] = ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        //  Chi
        for (j = 0; j < 25; j += 5) {
            bc[0] = st[j    ];
            bc[1] = st[j + 1];
            bc[2] = st[j + 2];
            bc[3] = st[j + 3];
            bc[4] = st[j + 4];
            st[j    ] ^= (~bc[1]) & bc[2];
            st[j + 1] ^= (~bc[2]) & bc[3];
            st[j + 2] ^= (~bc[3]) & bc[4];
            st[j + 3] ^= (~bc[4]) & bc[0];
            st[j + 4] ^= (~bc[0]) & bc[1];
        }

        //  Iota
        st[0] ^= keccakf_rndc[round];
    }
}


#define HASH_DATA_AREA	136

__device__ __forceinline__
void cn_keccak(const uint32_t * __restrict__ in, size_t inlen, uint64_t * __restrict__ md)
{
	uint64_t st[25];
	uint8_t temp[144];
	int i, rsiz, rsizw;
	size_t mdlen = sizeof(st);

	rsiz = sizeof(st) == mdlen ? HASH_DATA_AREA : 200 - 2 * mdlen;
	rsizw = rsiz / 8;

	memset(st, 0, sizeof(st));

	for ( ; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
			for (i = 0; i < rsizw; i++)
					st[i] ^= ((uint64_t *) in)[i];
			cn_keccakf(st, KECCAK_ROUNDS);
	}

	// last block and padding
	memcpy(temp, in, inlen);
	temp[inlen++] = 1;
	memset(temp + inlen, 0, rsiz - inlen);
	temp[rsiz - 1] |= 0x80;

	for (i = 0; i < rsizw; i++)
			st[i] ^= ((uint64_t *) temp)[i];

	cn_keccakf(st, KECCAK_ROUNDS);
	MEMCPY8(md, st, 25);

	return;
}
