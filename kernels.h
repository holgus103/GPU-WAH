/*
 * kernels.h
 *
 *  Created on: 4 mai 2018
 *      Author: holgus103
 */

#ifndef KERNELS_H_
#define KERNELS_H_
namespace regular_kernels{
	template<class T>
	__global__ void gpu_compressData(unsigned int* data_gpu, unsigned int* compressed_gpu, T* blockCounts_gpu, int dataSize);

	template<class T>
	__global__ void gpu_moveData(unsigned int* initialOutput_gpu, unsigned int* finalOutput_gpu, T* blockCounts_gpu);

	template<class T>
	__global__ void gpu_getCounts(unsigned int* data_gpu, T* counts_gpu, T dataSize);

	template<class T>
	__global__ void gpu_decompressWords(unsigned int* data_gpu, T* counts_gpu, unsigned int* result_gpu, T dataSize);

	template<class T>
	__global__ void gpu_mergeWords(unsigned int* result_gpu, unsigned int* finalOutput_gpu, T dataSize);
}


namespace no_sorting{
	__global__ void compressData(
			unsigned int* data_gpu,
			unsigned int* compressed_gpu,
			unsigned long long int* blockCounts_gpu,
			unsigned long long int* orderingArray_gpu,
			unsigned long long int* sizeCounter_gpu,
			int dataSize
			);

	__global__ void moveData(
			unsigned int* initialOutput_gpu,
			unsigned int* finalOutput_gpu,
			unsigned long long int* blockCounts_gpu
			);

	__global__ void getCounts(
			unsigned int* data_gpu,
			unsigned long long int* counts_gpu,
			unsigned long long int dataSize
			);

	__global__ void decompressWords(
			unsigned int* data_gpu,
			unsigned int* result_gpu,
			unsigned long long int* offsets,
			unsigned long long int* blockSizes,
			unsigned int blocks,
			unsigned long long int dataSize
			);

	__global__ void mergeWords(
			unsigned int* result_gpu,
			unsigned int* finalOutput_gpu,
			unsigned long long int dataSize
			);

	__global__ void reoderKernel(
			unsigned long long int* blockSizes,
			unsigned long long int* offsets,
			unsigned long long int* outputOffsets,
			unsigned long long int blockCount,
			unsigned int* data,
			unsigned long long int dataSize,
			unsigned int* output
	);
}

#endif /* KERNELS_H_ */
