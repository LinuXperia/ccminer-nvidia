#include <cuda_runtime.h>
#include <miner.h>
#include "cryptonight.h"

extern char *device_config[MAX_GPUS]; // -l 32x16
extern int device_bfactor[MAX_GPUS];

static __thread uint32_t cn_blocks;
static __thread uint32_t cn_threads;

// used for gpu intensity on algo init
static __thread bool gpu_init_shown = false;
#define gpulog_init(p,thr,fmt, ...) if (!gpu_init_shown) \
	gpulog(p, thr, fmt, ##__VA_ARGS__)


static bool init[MAX_GPUS] = { 0 };

nvid_ctx g_ctx;

extern "C" int scanhash_cryptonight_keva(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int _variant)
{
	int res = 0;
	uint32_t throughput = 0;

	uint32_t *ptarget = work->target;
	uint8_t *pdata = (uint8_t*) work->data;
	uint32_t *nonceptr = (uint32_t*) (&work->data[19]);
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = first_nonce;
	int dev_id = device_map[thr_id];

	const xmrig::Algo algorithm  = xmrig::CRYPTONIGHT;
	const xmrig::Variant variant = xmrig::VARIANT_2;

	if(opt_benchmark) {
		ptarget[7] = 0x00ff;
	}

	if(!init[thr_id])
	{
		int mem = cuda_available_memory(thr_id);
		int mul = device_sm[dev_id] >= 300 ? 4 : 1; // see cryptonight-core.cu
		cn_threads = device_sm[dev_id] >= 600 ? 16 : 8; // real TPB is x4 on SM3+
		cn_blocks = device_mpcount[dev_id] * 4;

		if (cn_blocks*cn_threads*2.2 > mem) cn_blocks = device_mpcount[dev_id] * 2;

		if (!opt_quiet)
			gpulog_init(LOG_INFO, thr_id, "%s, %d MB available, %hd SMX", device_name[dev_id],
				mem, device_mpcount[dev_id]);

		if (!device_config[thr_id] && strcmp(device_name[dev_id], "TITAN V") == 0) {
			device_config[thr_id] = strdup("80x24");
		}

		if (device_config[thr_id]) {
			int res = sscanf(device_config[thr_id], "%ux%u", &cn_blocks, &cn_threads);
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			gpulog_init(LOG_INFO, thr_id, "Using %ux%u(x%d) kernel launch config, %u threads",
				cn_blocks, cn_threads, mul, throughput);
		} else {
			throughput = cuda_default_throughput(thr_id, cn_blocks*cn_threads);
			if (throughput != cn_blocks*cn_threads && cn_threads) {
				cn_blocks = throughput / cn_threads;
				throughput = cn_threads * cn_blocks;
			}
			gpulog_init(LOG_INFO, thr_id, "%u threads (%g) with %u blocks",// of %ux%d",
				throughput, throughput2intensity(throughput), cn_blocks);//, cn_threads, mul);
		}

		if(sizeof(size_t) == 4 && throughput > UINT32_MAX / MEMORY) {
			gpulog(LOG_ERR, thr_id, "THE 32bit VERSION CAN'T ALLOCATE MORE THAN 4GB OF MEMORY!");
			gpulog(LOG_ERR, thr_id, "PLEASE REDUCE THE NUMBER OF THREADS OR BLOCKS");
			exit(1);
		}

		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		g_ctx.device_id        = dev_id;
		g_ctx.device_blocks    = cn_blocks;
		g_ctx.device_threads   = cn_threads;
		g_ctx.device_bfactor   = device_bfactor[thr_id];
		g_ctx.device_bsleep    = g_ctx.device_bfactor ? 100 : 0;
		g_ctx.syncMode         = 3; //cudaDeviceScheduleBlockingSync

		if (cuda_get_deviceinfo(&g_ctx, algorithm, false) != 0 || cryptonight_gpu_init(&g_ctx, algorithm) != 1) {
        printf("Setup failed for GPU %zu. Exitting.", thr_id);
        exit(1);
    }

		gpu_init_shown = true;
		init[thr_id] = true;
	}

	throughput = cn_blocks*cn_threads;

	uint64_t* target64 = (uint64_t*)(&ptarget[6]); // endanese?
	const uint32_t Htarg = ptarget[7];
	cryptonight_extra_cpu_set_data(&g_ctx, pdata, 20 * sizeof(uint32_t));
	do
	{
		uint32_t resNonces[10];
    uint32_t foundCount = 0;

    cryptonight_extra_cpu_prepare(&g_ctx, nonce, algorithm, variant);
    cryptonight_gpu_hash(&g_ctx, algorithm, variant, nonce);
    cryptonight_extra_cpu_final(&g_ctx, nonce, *target64, &foundCount, resNonces, algorithm);
		res = 0;
    for (size_t i = 0; i < foundCount; i++) {
				uint32_t vhash[8];
				uint32_t tempdata[20];
				uint32_t *tempnonceptr = (uint32_t*)(&tempdata[19]);
				memcpy(tempdata, pdata, 80);
				*tempnonceptr = resNonces[i];
				cryptonight_hash((const char*)tempdata, (char*)vhash, 80, variant);
				if(vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					work->nonces[i] = resNonces[i];
					work_set_target_ratio(work, vhash);
					res ++;
				} else if (!opt_quiet) {
						gpulog(LOG_WARNING, thr_id, "result for nonce %08x does not validate on CPU!", resNonces[i]);
				}

				if (res > 0) {
					goto done;
				}
    }

		*hashes_done = nonce - first_nonce + throughput;
		if ((uint64_t) throughput + nonce >= max_nonce - 127) {
			nonce = max_nonce;
			break;
		}

		nonce += throughput;
		gpulog(LOG_DEBUG, thr_id, "nonce %08x", nonce);

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + nonce);

done:
	gpulog(LOG_DEBUG, thr_id, "nonce %08x exit", nonce);
	work->valid_nonces = res;
	if (res == 1) {
		*nonceptr = work->nonces[0];
	} else if (res == 2) {
		*nonceptr = max(work->nonces[0], work->nonces[1]);
	} else {
		*nonceptr = nonce;
	}
	return res;
}

extern "C" int scanhash_cryptonight(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int variant)
{
	return false;
}

void free_cryptonight(int thr_id)
{
#if 0
	if (!init[thr_id])
		return;

	cudaFree(d_long_state[thr_id]);
	cudaFree(d_ctx_state[thr_id]);
	cudaFree(d_ctx_key1[thr_id]);
	cudaFree(d_ctx_key2[thr_id]);
	cudaFree(d_ctx_text[thr_id]);
	cudaFree(d_ctx_tweak[thr_id]);
	cudaFree(d_ctx_a[thr_id]);
	cudaFree(d_ctx_b[thr_id]);

	cryptonight_extra_free(thr_id);

	cudaDeviceSynchronize();

	init[thr_id] = false;
#endif
}
