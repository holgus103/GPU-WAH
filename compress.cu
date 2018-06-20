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

#define FREE_ALL \
	if(data_gpu) cudaFree(data_gpu);\
	if(compressed_gpu) cudaFree(compressed_gpu);\
	if(blockCounts_gpu) cudaFree(blockCounts_gpu);\
	if(finalOutput_gpu) cudaFree(finalOutput_gpu);\
	if(orderArray_gpu) cudaFree(orderArray_gpu);\
	if(sizeCounter_gpu) cudaFree(sizeCounter_gpu);


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
		unsigned int* outSize,
		unsigned int** orderingArray,
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

	int blockCount = dataSize / (31*32);

	if(dataSize % (31*32)> 0){
		blockCount++;
	}

	unsigned int *data_gpu, *compressed_gpu, *blockCounts_gpu, *finalOutput_gpu, *orderArray_gpu, *sizeCounter_gpu;

	// calculate max output size (one extra bit for every 31 bits)
	long long maxExpectedSize = 8*sizeof(int)*dataSize;
	if(maxExpectedSize % 31 > 0){
		maxExpectedSize /= 31;
		maxExpectedSize++;
	}
	else{
		maxExpectedSize /= 31;
	}

	dim3 blockSize = dim3(32, 32, 1);

	// allocate memory on the device
	if(cudaSuccess != cudaMalloc((void**)&sizeCounter_gpu, sizeof(int))){
		std::cout << "Could not allocate space for size counter" << std::endl;
		FREE_ALL
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&orderArray_gpu, blockCount * sizeof(int))){
		std::cout << "Could not allocate space for order array" << std::endl;
		FREE_ALL
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&data_gpu, dataSize * sizeof(int))){
		std::cout << "Could not allocate space for the data" << std::endl;
		FREE_ALL
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&compressed_gpu, maxExpectedSize * sizeof(int))){
		std::cout << "Could not allocate space for the compressed output" << std::endl;
		FREE_ALL
		return NULL;
	}
	if(cudaSuccess != cudaMalloc((void**)&blockCounts_gpu, blockCount* sizeof(int))){
		std::cout << "Could not allocate space for the block sizes" << std::endl;
		FREE_ALL
		return NULL;
	}

	// copy input
	if(cudaSuccess != cudaMemcpy(data_gpu, data_cpu, dataSize*sizeof(int), cudaMemcpyHostToDevice)){
		std::cout << "Could not copy input" << std::endl;
		FREE_ALL
		return NULL;
	}

	// get transfer time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&transferToDeviceTime, start,stop);

	// restart time measuring
	cudaEventCreate(&start);
	cudaEventRecord(start,0);

	// call compression kernel
	compressData<<<blockCount,blockSize>>>(data_gpu, compressed_gpu, blockCounts_gpu, orderArray_gpu, sizeCounter_gpu, dataSize);

	// remove unnecessary data
	cudaFree((void*)data_gpu);


	unsigned int* orderArray = (unsigned int*)malloc(sizeof(int) * blockCount);
	(*orderingArray) = orderArray;
	if(cudaSuccess != cudaMemcpy(&orderArray, orderArray_gpu, blockCount * sizeof(int), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy last block count" << std::endl;
		FREE_ALL
		return NULL;
	}

	unsigned int outputSize = 0;;

	if(cudaSuccess != cudaMemcpy(&outputSize, sizeCounter_gpu, sizeof(int), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy last block offset" << std::endl;
		FREE_ALL
		return NULL;
	}

	SAFE_ASSIGN(outSize, outputSize)
	if(cudaSuccess != cudaMalloc((void**)&finalOutput_gpu, sizeof(int) * outputSize)){
		std::cout << "Could not allocate final Output" << std::endl;
		FREE_ALL
		return NULL;
	}

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
	if(cudaSuccess != cudaMemcpy((void*)compressed_cpu, (void*)finalOutput_gpu, outputSize * sizeof(int), cudaMemcpyDeviceToHost)){
		std::cout << "Could not copy final output" << std::endl;
	}

	// free gpu memory
	cudaFree((void*)compressed_gpu);
	cudaFree((void*)blockCounts_gpu);
	cudaFree((void*)finalOutput_gpu);
	cudaFree((void*)orderArray_gpu);

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



