#pragma once

#ifndef _WIN32
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <string>
#endif

/** execute and check a CUDA api command
*
* @param id gpu id (thread id)
* @param ... CUDA api command
*/
#ifdef _WIN32
#define CUDA_CHECK(id, ...) {                                                                             \
cudaError_t error = __VA_ARGS__;                                                                          \
    if(error!=cudaSuccess){																				  \
		printf("[CUDA] Error gpu: %d, function: %s,  line: %d, error: %s \n",                             \
			id, __FUNCTION__, __LINE__, cudaGetErrorString(error));                                       \
	}                                                                                                     \
}                                                                                                         \
( (void) 0 )
#else
#define CUDA_CHECK(id, ...) {                                                                             \
    cudaError_t error = __VA_ARGS__;                                                                      \
    if(error!=cudaSuccess){                                                                               \
        std::cerr << "[CUDA] Error gpu " << id << ": <" << __FUNCTION__ << ">:" << __LINE__ << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
        throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error))); \
    }                                                                                                     \
}                                                                                                         \
( (void) 0 )
#endif

/** execute and check a CUDA kernel
*
* @param id gpu id (thread id)
* @param ... CUDA kernel call
*/
#define CUDA_CHECK_KERNEL(id, ...)      \
    __VA_ARGS__;                        \
    CUDA_CHECK(id, cudaGetLastError())
