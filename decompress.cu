#include "decompress.h"
#include "kernels.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "timeMeasuring.h"

unsigned int* decompress(
		unsigned int* data,
		unsigned long long int dataSize,
		unsigned long long int* outSize,
		float* pTransferToDeviceTime,
		float* pCompressionTime,
		float* ptranserFromDeviceTime){

	// times to be measured
	float transferToDeviceTime;
	float compressionTime;
	float transferFromDeviceTime;

	// start measuring time
	CREATE_TIMER
	START_TIMER

	unsigned int *data_gpu, *result_gpu, *finalOutput_gpu, *output_cpu;
	unsigned long long int* counts_gpu;
	unsigned long long int blockCount = dataSize / 1024;

	if(dataSize % 1024 > 0){
		blockCount++;
	}
	cudaMalloc((void**)&data_gpu, sizeof(int)*dataSize);
	cudaMalloc((void**)&counts_gpu, sizeof(int)*dataSize);
	cudaMemcpy(data_gpu, data, sizeof(int)*dataSize, cudaMemcpyHostToDevice);

	STOP_TIMER
	GET_RESULT(transferToDeviceTime)
	START_TIMER
//	counts_cpu = (unsigned int*) malloc(sizeof(int)*dataSize);
	dim3 blockDim(32, 32);
	// get blocked sizes
	getCounts<<<blockCount,blockDim>>>(data_gpu, counts_gpu, dataSize);
	unsigned int lastBlockSize;
	cudaMemcpy(&lastBlockSize, counts_gpu  + (dataSize - 1), sizeof(int), cudaMemcpyDeviceToHost);
	// scan block sizes
	thrust::device_ptr<unsigned long long int> countsPtr(counts_gpu);
	// get counts
	thrust::exclusive_scan(countsPtr, countsPtr + dataSize, countsPtr);
	unsigned int lastOffset;
//	thrust::inclusive_scan(counts_cpu, counts_cpu + dataSize, counts_cpu);
	cudaMemcpy(&lastOffset, counts_gpu + (dataSize - 1), sizeof(int), cudaMemcpyDeviceToHost);
	int outputSize = lastBlockSize + lastOffset;
	int realSize = 31*outputSize;

	if(realSize % 32 > 0){
		realSize /=32;
		realSize++;
	}
	else{
		realSize /=32;
	}
	SAFE_ASSIGN(outSize, realSize);
//	free(counts_cpu);
	cudaMalloc((void**)&result_gpu, sizeof(int) * outputSize);

	decompressWords<<<blockCount,blockDim>>>(data_gpu, counts_gpu, result_gpu, dataSize);
	cudaFree(data_gpu);
	cudaFree(counts_gpu);

	blockCount = outputSize / 1024;
	if(dataSize % 1024 > 0){
		blockCount++;
	}

	cudaMalloc((void**)&finalOutput_gpu, sizeof(int)*outputSize);
	mergeWords<<<blockCount,blockDim>>>(result_gpu, finalOutput_gpu, outputSize);
	cudaFree(result_gpu);

	STOP_TIMER
	GET_RESULT(compressionTime)
	START_TIMER

	output_cpu = (unsigned int*)malloc(sizeof(int) * outputSize);
	cudaMemcpy(output_cpu, finalOutput_gpu, sizeof(int) * outputSize, cudaMemcpyDeviceToHost);
	cudaFree(finalOutput_gpu);
	STOP_TIMER
	GET_RESULT(transferFromDeviceTime)

	SAFE_ASSIGN(pCompressionTime, compressionTime);
	SAFE_ASSIGN(pTransferToDeviceTime, transferToDeviceTime);
	SAFE_ASSIGN(ptranserFromDeviceTime, transferFromDeviceTime);

	return output_cpu;
}

unsigned int* reorder(
		unsigned long long int* blockSizes,
		unsigned long long int* offsets,
		unsigned long long int blockCount,
		unsigned int* data,
		unsigned long long int dataSize,
		float* pTransferToDeviceTime,
		float* pReoderingTime,
		float* ptranserFromDeviceTime
		){

	// times to be measured
	float transferToDeviceTime;
	float reorderingTime;
	float transferFromDeviceTime;

	CREATE_TIMER
	START_TIMER

	unsigned int *data_gpu, *output_gpu;
	unsigned long long int* blockSizes_gpu, *offsets_gpu, *outputOffsets_gpu;
	// allocate gpu memory
	cudaMalloc((void**)&blockSizes_gpu, sizeof(int)*blockCount);
	cudaMalloc((void**)&offsets_gpu, sizeof(int)*blockCount);
	cudaMalloc((void**)&data_gpu, sizeof(int)*dataSize);
	cudaMalloc((void**)&output_gpu, sizeof(int)*dataSize);
	cudaMalloc((void**)&outputOffsets_gpu, sizeof(int)*dataSize);

	cudaMemcpy(blockSizes_gpu, blockSizes, sizeof(int) * blockCount, cudaMemcpyHostToDevice);
	cudaMemcpy(offsets_gpu, offsets, sizeof(int) * blockCount, cudaMemcpyHostToDevice);
	cudaMemcpy(data_gpu, data, sizeof(int) * dataSize, cudaMemcpyHostToDevice);

	STOP_TIMER
	GET_RESULT(transferToDeviceTime)

	START_TIMER

	thrust::device_ptr<unsigned long long int> pBlockSizes(blockSizes_gpu);
	thrust::device_ptr<unsigned long long int> pOutputOffets(outputOffsets_gpu);

	thrust::exclusive_scan(pBlockSizes, pBlockSizes + blockCount, pOutputOffets);

	int b = dataSize / 1024;

	if(dataSize % 1024 > 0){
		b++;
	}

	reoderKernel<<<b, dim3(32, 32)>>>(blockSizes_gpu, offsets_gpu, outputOffsets_gpu, blockCount, data_gpu, dataSize, output_gpu);

	STOP_TIMER
	GET_RESULT(reorderingTime)

	START_TIMER

	unsigned int* output = (unsigned int*) malloc(sizeof(int) * dataSize);
	cudaMemcpy(output, output_gpu, sizeof(int) * dataSize, cudaMemcpyDeviceToHost);
	cudaFree(blockSizes_gpu);
	cudaFree(offsets_gpu);
	cudaFree(data_gpu);
	cudaFree(outputOffsets_gpu);
	cudaFree(output_gpu);


	STOP_TIMER
	GET_RESULT(transferFromDeviceTime)

	SAFE_ASSIGN(pReoderingTime, reorderingTime);
	SAFE_ASSIGN(pTransferToDeviceTime, transferToDeviceTime);
	SAFE_ASSIGN(ptranserFromDeviceTime, transferFromDeviceTime);

	return output;

}
