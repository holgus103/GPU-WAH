/*
 * kernels.h
 *
 *  Created on: 4 mai 2018
 *      Author: holgus103
 */

#ifndef KERNELS_H_
#define KERNELS_H_


__global__ void compressData(
		unsigned int* data_gpu,
		unsigned int* compressed_gpu,
		unsigned int* blockCounts_gpu,
		unsigned int* orderingArray_gpu,
		unsigned int* sizeCounter_gpu,
		int dataSize
		);

__global__ void moveData(
		unsigned int* initialOutput_gpu,
		unsigned int* finalOutput_gpu,
		unsigned int* blockCounts_gpu
		);

__global__ void getCounts(
		unsigned int* data_gpu,
		unsigned int* counts_gpu,
		int dataSize
		);

__global__ void decompressWords(
		unsigned int* data_gpu,
		unsigned int* counts_gpu,
		unsigned int* result_gpu,
		int dataSize
		);

__global__ void mergeWords(
		unsigned int* result_gpu,
		unsigned int* finalOutput_gpu,
		int dataSize
		);

#endif /* KERNELS_H_ */
