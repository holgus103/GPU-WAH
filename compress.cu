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
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


struct is_zero
{
	 __host__ __device__
	 bool operator()(const int x)
	 {
	   return x == 0;
	 }
};


// datasize is in integers!
unsigned int* compress(unsigned int* data_cpu, unsigned int dataSize){
	int blockCount = dataSize / 1024;

	if(dataSize % 1024 > 0){
		blockCount++;
	}

	unsigned int *data_gpu, *compressed_gpu, *blockCounts_gpu, *finalOutput_gpu;

	// calculate max output size (one extra bit for every 31 bits)
	long long maxExpectedSize = 8*sizeof(int)*dataSize;
	if(maxExpectedSize % 31 > 0){
		maxExpectedSize /= 31;
		maxExpectedSize++;
	}
	else{
		maxExpectedSize /= 31;
	}

	maxExpectedSize *= 8*sizeof(int);

	dim3 blockSize = dim3(32, 32, 1);
	// allocate memory on the device
	cudaMalloc((void**)&data_gpu, dataSize * sizeof(int));
	cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int));
	cudaMalloc((void**)&blockCounts_gpu, blockCount* sizeof(int));
	// copy input
	cudaMemcpy(data_gpu, data_cpu, dataSize*sizeof(int), cudaMemcpyHostToDevice);

	// call compression kernel
	compressData<<<blockCount,blockSize>>>(data_gpu, compressed_gpu, blockCounts_gpu, dataSize);
	// remove unnecessary data
	cudaFree((void*)data_gpu);
	thrust::device_ptr<unsigned int> blockCountsPtr(blockCounts_gpu);
	unsigned int* wordNumbers = (unsigned int*)malloc(sizeof(int)*blockCount);
	cudaMemcpy(wordNumbers, blockCounts_gpu, sizeof(int) *blockCount, cudaMemcpyDeviceToHost);
	thrust::exclusive_scan(blockCountsPtr, blockCountsPtr + blockCount, blockCountsPtr);
	thrust::inclusive_scan(wordNumbers, wordNumbers + blockCount, wordNumbers);
	cudaMalloc((void**)&finalOutput_gpu, sizeof(int) * wordNumbers[blockCount-1]);
	// call merge kernel
	moveData<<<blockCount, blockSize>>>(compressed_gpu, finalOutput_gpu, blockCounts_gpu);
	// allocate memory for results
	unsigned int* compressed_cpu = (unsigned int*)malloc(sizeof(int)*wordNumbers[blockCount-1]);
	// copy compressed data
	cudaMemcpy((void*)compressed_cpu, (void*)finalOutput_gpu, wordNumbers[blockCount-1] * sizeof(int), cudaMemcpyDeviceToHost);

	// free gpu memory

	cudaFree((void*)compressed_gpu);
	cudaFree((void*)blockCounts_gpu);
	cudaFree((void*)finalOutput_gpu);
	free(wordNumbers);
	return compressed_cpu;
}



