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
#include "timeMeasuring.h"


struct is_zero
{
	 __host__ __device__
	 bool operator()(const int x)
	 {
	   return x == 0;
	 }
};



/*
 * Host function performing compression
 *
 * Parameters:
 * data_cpu - host pointer to data to be compressed
 * dataSize - size of the data in integers
 * outSize - size of the output in integers
 * pTransferToDeviceTime - pointer to the output parameter storing the transfer time to the device
 * pCompressionTime - pointer to the output parameter storing the compression time
 * ptranserFromDeviceTime - pointer to the output parameter storing the transfer time from the device
 */
template<class T>
unsigned int* compress(
		unsigned int* data_cpu,
		T dataSize,
		T* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime){

// -- Variable initialization --

	// times to be measured
	float transferToDeviceTime;
	float compressionTime;
	float transferFromDeviceTime;

	// start measuring time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// calculate the number of blocks necessary
	int blockCount = dataSize / (31*32);

	// if not divisible, add another additional block
	if(dataSize % (31*32)> 0){
		blockCount++;
	}

	// device data pointers
	unsigned int *data_gpu, *compressed_gpu, *finalOutput_gpu;
	T* blockCounts_gpu;

	// calculate max output size (one extra bit for every 31 bits)
	unsigned long long int maxExpectedSize = 8*sizeof(int)*dataSize;
	if(maxExpectedSize % 31 > 0){
		maxExpectedSize /= 31;
		maxExpectedSize++;
	}
	else{
		maxExpectedSize /= 31;
	}

	// initialize block dimensions
	dim3 blockSize = dim3(32, 32, 1);

// -- Memory allocation --

	// allocate memory on the device
	if(cudaSuccess != cudaMalloc((void**)&data_gpu, dataSize * sizeof(int))){
		std::cout << "Could not allocate space for the data" << std::endl;
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int))){
		std::cout << "Could not allocate space for the compressed output" << std::endl;
		cudaFree(data_gpu);
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&blockCounts_gpu, blockCount* sizeof(T))){
		std::cout << "Could not allocate space for the block sizes" << std::endl;
		cudaFree(data_gpu);
		cudaFree(compressed_gpu);
		return NULL;
	}

// -- Data transfer --

	// copy input
	if(cudaSuccess != cudaMemcpy(data_gpu, data_cpu, dataSize*sizeof(int), cudaMemcpyHostToDevice)){
		std::cout << "Could not copy input" << std::endl;
		cudaFree(data_gpu);
		cudaFree(compressed_gpu);
		cudaFree(blockCounts_gpu);
		return NULL;
	}

	// get transfer time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transferToDeviceTime, start,stop);

// -- Data compression --

	// restart time measuring
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// call compression kernel, merges words within a block
	gpu_compressData<T><<<blockCount,blockSize>>>(data_gpu, compressed_gpu, blockCounts_gpu, dataSize);

	// remove unnecessary data
	cudaFree((void*)data_gpu);
	thrust::device_ptr<T> blockCountsPtr(blockCounts_gpu);


	T lastWordNumber;

	// get the size of the last block
	if(cudaSuccess != cudaMemcpy(&lastWordNumber, blockCounts_gpu + (blockCount - 1), sizeof(T), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy last block count" << std::endl;
		cudaFree(compressed_gpu);
		cudaFree(blockCounts_gpu);
		return NULL;
	}

	thrust::exclusive_scan(blockCountsPtr, blockCountsPtr + blockCount, blockCountsPtr);
	T  lastBlockOffset;

	// get the offset of the last block
	if(cudaSuccess != cudaMemcpy(&lastBlockOffset, blockCounts_gpu + (blockCount - 1), sizeof(T), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy last block offset" << std::endl;
		cudaFree(compressed_gpu);
		cudaFree(blockCounts_gpu);
		return NULL;
	}

	T outputSize = lastBlockOffset + lastWordNumber;
	SAFE_ASSIGN(outSize, outputSize)
	if(cudaSuccess != cudaMalloc((void**)&finalOutput_gpu, sizeof(int) * outputSize)){
		std::cout << "Could not allocate final Output" << std::endl;
		cudaFree(compressed_gpu);
		cudaFree(blockCounts_gpu);
		return NULL;
	}
	// call merge kernel
	gpu_moveData<T><<<blockCount, blockSize>>>(compressed_gpu, finalOutput_gpu, blockCounts_gpu);

	// get compression time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&compressionTime, start,stop);

// -- Move decompressed data from device to host --

	// restart time measuring
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// allocate memory for results
	unsigned int* compressed_cpu = (unsigned int*)malloc(sizeof(int)* outputSize);

	// copy compressed data
	if(cudaSuccess != cudaMemcpy((void*)compressed_cpu, (void*)finalOutput_gpu, outputSize * sizeof(int), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy final output" << std::endl;
	}

// -- Cleanup --

	// free gpu memory
	cudaFree((void*)compressed_gpu);
	cudaFree((void*)blockCounts_gpu);
	cudaFree((void*)finalOutput_gpu);


// -- Get stats and save them to output parameters --

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


template unsigned int* compress<unsigned long long int>(
		unsigned int* data_cpu,
		unsigned long long int dataSize,
		unsigned long long int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

template unsigned int* compress<unsigned int>(
		unsigned int* data_cpu,
		unsigned int dataSize,
		unsigned int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

