/*
 * kernels.h
 *
 *  Created on: 4 mai 2018
 *      Author: holgus103
 */

#ifndef KERNELS_H_
#define KERNELS_H_

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

#endif /* KERNELS_H_ */
