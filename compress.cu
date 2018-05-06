/*
 * compress.cu
 *
 *  Created on: 6 mai 2018
 *      Author: holgus103
 */
#include "compress.h"
#include "kernels.h"
#include <cuda.h>
#include <math.h>
#include <cuda_device_runtime_api.h>

int* compress(int* data_cpu, int dataSize){
	int * data_gpu, compressed_gpu;

	// allocate memory for results
	int * compressed_cpu = malloc(sizeof(int)*dataSize);

	// calculate max output size (one extra bit for every 31 bits)
	long long maxExpectedSize = 8*sizeof(int)*dataSize;
	maxExpectedSize *= 32;
	maxExpectedSize /= 31;
	maxExpectedSize /= 8*sizeof(int);

	// increment in case it got rounded
	maxExpectedSize++;

	// allocate memory on the device
	cudaMalloc((void**)&data_gpu, dataSize * sizeof(int));
	cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int));

	// copy input
	cudaMemcpy(data_gpu, data, dataSize*sizeof(int), cudaMemcpyHostToDevice);

	// call compression kernel
	compressData<<<1,1>>>(data_gpu, compressed_gpu);

	// copy compressed data
	cudaMemcpy(compressed_cpu, compressed_gpu, maxExpectedSize * sizeof(int), cudaMemcpyDeviceToHost);

	return compressed_cpu;
}



