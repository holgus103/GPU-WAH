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


struct is_zero
{
	 __host__ __device__
	 bool operator()(const int x)
	 {
	   return x == 0;
	 }
};


// datasize is in bytes!
unsigned int* compress(unsigned int* data_cpu, unsigned int dataSize){
	int blockCount = dataSize / (1024 *sizeof(int));

	if(dataSize % (1024*sizeof(int)) > 0){
		blockCount++;
	}

	unsigned int *data_gpu, *compressed_gpu;

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
	if(maxExpectedSize)

	dim3 dimBlock(32, 32);
	// allocate memory for results
	unsigned int* compressed_cpu = (unsigned int*)malloc(sizeof(int)*maxExpectedSize);
	// allocate memory on the device
	cudaMalloc((void**)&data_gpu, dataSize * sizeof(int));
	cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int));

	// copy input
	cudaMemcpy(data_gpu, data_cpu, dataSize*sizeof(int), cudaMemcpyHostToDevice);

	// call compression kernel
	compressData<<<blockCount,dimBlock>>>(data_gpu, compressed_gpu, dataSize);
//	compressed_gpu = thrust::remove_if(compressed_gpu, compressed_gpu + maxExpectedSize, is_zero());

	// copy compressed data
	cudaMemcpy((void*)compressed_cpu, (void*)compressed_gpu, maxExpectedSize * sizeof(int), cudaMemcpyDeviceToHost);
	
	// free gpu memory
	cudaFree((void*)data_gpu);
	cudaFree((void*)compressed_gpu);

	return compressed_cpu;
}



