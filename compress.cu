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
#include <time.h>
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
unsigned int* compress(
		unsigned int* data_cpu,
		unsigned int dataSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime){

	// times to be measured
	float transferToDeviceTime;
	float compressionTime;
	float transferFromDeviceTime;

	// start measuring time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

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

	// get transfer time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transferToDeviceTime, start,stop);

	// restart time measuring
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// call compression kernel
	compressData<<<blockCount,blockSize>>>(data_gpu, compressed_gpu, blockCounts_gpu, dataSize);

	// remove unnecessary data
	cudaFree((void*)data_gpu);
	thrust::device_ptr<unsigned int> blockCountsPtr(blockCounts_gpu);


	unsigned int lastWordNumber;
	cudaMemcpy(&lastWordNumber, blockCounts_gpu + (blockCount - 1), sizeof(int), cudaMemcpyDeviceToHost);
	thrust::exclusive_scan(blockCountsPtr, blockCountsPtr + blockCount, blockCountsPtr);
	unsigned int lastBlockOffset;
	cudaMemcpy(&lastBlockOffset, blockCounts_gpu + (blockCount - 1), sizeof(int), cudaMemcpyDeviceToHost);

	int outputSize = lastBlockOffset + lastWordNumber;
	cudaMalloc((void**)&finalOutput_gpu, sizeof(int) * outputSize);
	// call merge kernel
	moveData<<<blockCount, blockSize>>>(compressed_gpu, finalOutput_gpu, blockCounts_gpu);

	// get compression time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&compressionTime, start,stop);

	// restart time measuring
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// allocate memory for results
	unsigned int* compressed_cpu = (unsigned int*)malloc(sizeof(int)* outputSize);
	// copy compressed data
	cudaMemcpy((void*)compressed_cpu, (void*)finalOutput_gpu, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

	// free gpu memory
	cudaFree((void*)compressed_gpu);
	cudaFree((void*)blockCounts_gpu);
	cudaFree((void*)finalOutput_gpu);

	// get transfer time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transferFromDeviceTime, start,stop);

	// write results to pointers if specified
	if(pCompressionTime != NULL) (*pCompressionTime) = compressionTime;
	if(pTransferToDeviceTime != NULL) (*pTransferToDeviceTime) = transferToDeviceTime;
	if(ptranserFromDeviceTime != NULL) (*ptranserFromDeviceTime) = transferFromDeviceTime;
	return compressed_cpu;
}



