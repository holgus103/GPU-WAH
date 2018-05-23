#include "decompress.h"
#include "kernels.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

unsigned int* decompress(unsigned int* data, unsigned int dataSize){
	unsigned int *data_gpu, *counts_gpu, *result_gpu, *finalOutput_gpu, *counts_cpu, *output_cpu;
	int blockCount = dataSize / 1024;

	if(dataSize % 1024 > 0){
		blockCount++;
	}
	cudaMalloc((void**)&data_gpu, sizeof(int)*dataSize);
	cudaMalloc((void**)&counts_gpu, sizeof(int)*dataSize);
	cudaMemcpy(data_gpu, data, sizeof(int)*dataSize, cudaMemcpyHostToDevice);
	counts_cpu = (unsigned int*) malloc(sizeof(int)*dataSize);
	dim3 blockDim(32, 32);
	// get blocked sizes
	getCounts<<<blockCount,blockDim>>>(data_gpu, counts_gpu, dataSize);
	cudaMemcpy(counts_cpu, counts_gpu, sizeof(int)*dataSize, cudaMemcpyDeviceToHost);
	// scan block sizes
	thrust::device_ptr<unsigned int> countsPtr(counts_gpu);
	// get counts
	thrust::exclusive_scan(countsPtr, countsPtr + dataSize, countsPtr);
	thrust::inclusive_scan(counts_cpu, counts_cpu + dataSize, counts_cpu);
	int outputSize = counts_cpu[dataSize - 1];
	free(counts_cpu);
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

	output_cpu = (unsigned int*)malloc(sizeof(int) * outputSize);
	cudaMemcpy(output_cpu, finalOutput_gpu, sizeof(int) * outputSize, cudaMemcpyDeviceToHost);
	cudaFree(finalOutput_gpu);

	return output_cpu;
}
