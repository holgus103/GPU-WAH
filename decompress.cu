#include "decompress.h"
#include "kernels.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "timeMeasuring.h"

/*
 * Host function performing compression
 *
 * Parameters:
 * data - host pointer to data to be compressed
 * dataSize - size of the data in integers
 * outSize - size of the output in integers
 * pTransferToDeviceTime - pointer to the output parameter storing the transfer time to the device
 * pCompressionTime - pointer to the output parameter storing the decompression time
 * ptranserFromDeviceTime - pointer to the output parameter storing the transfer time from the device
 */
template<class T>
unsigned int* decompress(
		unsigned int* data,
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
	CREATE_TIMER
	START_TIMER

	// data pointers
	unsigned int *data_gpu, *result_gpu, *finalOutput_gpu, *output_cpu;
	T* counts_gpu;
	int blockCount = dataSize / 1024;

	if(dataSize % 1024 > 0){
		blockCount++;
	}

//	-- Memory allocation --

	cudaMalloc((void**)&data_gpu, sizeof(int)*dataSize);
	cudaMalloc((void**)&counts_gpu, sizeof(T)*dataSize);

// -- Data transfer --
	cudaMemcpy(data_gpu, data, sizeof(int)*dataSize, cudaMemcpyHostToDevice);

	STOP_TIMER
	GET_RESULT(transferToDeviceTime)
	START_TIMER

//	counts_cpu = (unsigned int*) malloc(sizeof(int)*dataSize);

	// initialize block dimensions
	dim3 blockDim(32, 32);

// -- Decompress data --

	// get blocked sizes of all words in the compressed file
	getCounts<T><<<blockCount,blockDim>>>(data_gpu, counts_gpu, dataSize);

	// size of the last block
	T lastBlockSize;

	// copy the size of the last block
	cudaMemcpy(&lastBlockSize, counts_gpu  + (dataSize - 1), sizeof(T), cudaMemcpyDeviceToHost);

	thrust::device_ptr<T> countsPtr(counts_gpu);
	// scan block sizes and save offsets
	thrust::exclusive_scan(countsPtr, countsPtr + dataSize, countsPtr);
	// offset of the last block
	T lastOffset;
	// copy the offset of the last block
	cudaMemcpy(&lastOffset, counts_gpu + (dataSize - 1), sizeof(T), cudaMemcpyDeviceToHost);
	// calculate output size in 31-bit words
	unsigned long long int outputSize = lastBlockSize + lastOffset;
	// calculate the real output size in integers
	unsigned long long int realSize = 31*outputSize;

	if(realSize % 32 > 0){
		realSize /=32;
		realSize++;
	}
	else{
		realSize /=32;
	}
	SAFE_ASSIGN(outSize, realSize);
//	free(counts_cpu);

	// allocate memory for the result
	cudaMalloc((void**)&result_gpu, sizeof(int) * outputSize);

	// decompress the words
	decompressWords<T><<<blockCount,blockDim>>>(data_gpu, counts_gpu, result_gpu, dataSize);

	// free data and block offsets
	cudaFree(data_gpu);
	cudaFree(counts_gpu);

	// calculate block count
	blockCount = outputSize / 1024;
	if(dataSize % 1024 > 0){
		blockCount++;
	}

	// allocate memory for final output data
	cudaMalloc((void**)&finalOutput_gpu, sizeof(int)*outputSize);
	// convert from 31 bit words to integers
	mergeWords<T><<<blockCount,blockDim>>>(result_gpu, finalOutput_gpu, outputSize);

// -- Cleanup after decompression --

	// free incomplete array
	cudaFree(result_gpu);

	STOP_TIMER
	GET_RESULT(compressionTime)
	START_TIMER

	// allocate host array and copy final result
	output_cpu = (unsigned int*)malloc(sizeof(int) * outputSize);
	cudaMemcpy(output_cpu, finalOutput_gpu, sizeof(int) * outputSize, cudaMemcpyDeviceToHost);

	// free device result array
	cudaFree(finalOutput_gpu);

	STOP_TIMER
	GET_RESULT(transferFromDeviceTime)

	SAFE_ASSIGN(pCompressionTime, compressionTime);
	SAFE_ASSIGN(pTransferToDeviceTime, transferToDeviceTime);
	SAFE_ASSIGN(ptranserFromDeviceTime, transferFromDeviceTime);

	return output_cpu;
}

template unsigned int* decompress<unsigned long long int>(
		unsigned int* data,
		unsigned long long int dataSize,
		unsigned long long int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);

template unsigned int* decompress<unsigned int>(
		unsigned int* data,
		unsigned int dataSize,
		unsigned int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime);
