/*
 * compress.cu
 *
 *  Created on: 6 mai 2018
 *      Author: holgus103
 */
#include "compress.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdlib.h>  

int* compress(int* data_cpu, int dataSize){
	int* data_gpu,* compressed_gpu, int* warpInfo_gpu;

	// calculate max output size (one extra bit for every 31 bits)
	long long maxExpectedSize = 8*sizeof(int)*dataSize;
	maxExpectedSize *= 32;
	maxExpectedSize /= 31;
	maxExpectedSize /= 8*sizeof(int);

	// increment in case it got rounded
	maxExpectedSize++;

	// allocate memory for results
	int* compressed_cpu = (int*)malloc(sizeof(int)*maxExpectedSize);

	// allocate memory on the device
	cudaMalloc((void**)&data_gpu, dataSize * sizeof(int));
	cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int));
	cudaMalloc((void**)&warpInfo_gpu, 1 * sizeof(int));

	// copy input
	cudaMemcpy(data_gpu, data_cpu, dataSize*sizeof(int), cudaMemcpyHostToDevice);

	// call compression kernel
	compressData<<<1,32>>>(data_gpu, compressed_gpu, warpInfo_gpu);

	// copy compressed data
	cudaMemcpy((void*)compressed_cpu, (void*)compressed_gpu, maxExpectedSize * sizeof(int), cudaMemcpyDeviceToHost);
	
	// free gpu memory
	cudaFree((void*)data_gpu);
	cudaFree((void*)compressed_gpu);
	cudaFree((void*)warpInfo_gpu);

	return compressed_cpu;
}



