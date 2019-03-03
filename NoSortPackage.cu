/*
 * NoSortPackage.cpp
 *
 *  Created on: Feb 13, 2019
 *      Author: holgus103
 */

#include "NoSortPackage.h"
#include "CompressedPackage.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include "timeMeasuring.h"

template<class T>
NoSortPackage<T>::NoSortPackage() {

}

template<class T>
NoSortPackage<T>::~NoSortPackage() {

}


template<class T>
void NoSortPackage<T>::c_allocateMemory(){
	CompressedPackage<T>::c_allocateMemory();
	// allocate memory on the device
	if(cudaSuccess != cudaMalloc((void**)&(this->sizeCounter_gpu), sizeof(T))){
		std::cout << "Could not allocate space for size counter" << std::endl;
		this->c_cleanup();
		return;
	}
	if(cudaSuccess != cudaMalloc((void**)&(this->orderingArray_gpu), this->blockCount * sizeof(T))){
		std::cout << "Could not allocate space for order array" << std::endl;
		this->c_cleanup();
		return;
	}

}


template<class T>
void NoSortPackage<T>::c_runAlgorithm(){
		dim3 blockSize = dim3(32, 32, 1);
		// call compression kernel
		no_sorting::compressData<<<this->blockCount,blockSize>>>(this->data_gpu, this->compressed_gpu, this->blockCounts_gpu, this->orderingArray_gpu, this->sizeCounter_gpu, this->size);

		// remove unnecessary data
		cudaFree((void*)this->data_gpu);

		// allocate memory for block sizes
		this->blockSizes = (T*)malloc(sizeof(T) *this->blockCount);

		// copy block sizes
		if(cudaSuccess != cudaMemcpy(this->blockSizes, this->blockCounts_gpu, this->blockCount * sizeof(T), cudaMemcpyDeviceToHost)){
			std::cout << "Could not copy last block counts" << std::endl;
			this->c_cleanup();
			return;
		}

		// allocate ordering array
		this->orderingArray = (T*) malloc(sizeof(T) * this->blockCount);

		// copy ordering array
		if(cudaSuccess != cudaMemcpy(this->orderingArray, this->orderingArray_gpu, this->blockCount * sizeof(T), cudaMemcpyDeviceToHost)){
			std::cout << "Could not copy ordering array" << std::endl;
			this->c_cleanup();
			return;
		}

		unsigned long long int outputSize = 0;;

		if(cudaSuccess != cudaMemcpy(&outputSize, this->sizeCounter_gpu, sizeof(T), cudaMemcpyDeviceToHost)){
			std::cout << "Could not copy last block offset" << std::endl;
			this->c_cleanup();
			return;
		}
		this->compressedSize = outputSize;
		this->finalOutput_gpu = this->compressed_gpu;
}


template<class T>
void NoSortPackage<T>::c_cleanup(){
	if(this->data_gpu) cudaFree(this->data_gpu);
	if(this->compressed_gpu) cudaFree(this->compressed_gpu);
	if(this->blockCounts_gpu) cudaFree(this->blockCounts_gpu);
	if(this->orderingArray_gpu) cudaFree(this->orderingArray_gpu);
	if(this->sizeCounter_gpu) cudaFree(this->sizeCounter_gpu);
}

template<class T>
void NoSortPackage<T>::d_allocateMemory(){
	CompressedPackage<T>::d_allocateMemory();
	if(cudaSuccess != cudaMalloc((void**)&(this->orderingArray_gpu), sizeof(T)* this->blockCount)){
		std::cout << "Decomp: Could not allocate space for offset array" << std::endl;
	}
}

template<class T>
void NoSortPackage<T>::d_copyToDevice(){
	// -- Data transfer --
	CompressedPackage<T>::d_copyToDevice();
	cudaMemcpy(this->orderingArray_gpu, this->orderingArray, sizeof(T)*this->blockCount, cudaMemcpyHostToDevice);
	cudaMemcpy(this->counts_gpu, this->blockSizes, sizeof(T) * this->blockCount, cudaMemcpyHostToDevice);

}

template<class T>
void NoSortPackage<T>::d_runAlgorithm(){
	dim3 blockDim(32, 32);
	// get blocked sizes
	unsigned long long int outputSize = 1024*this->blockCount;
	unsigned long long int realSize = 31*outputSize;

	if(realSize % 32 > 0){
		realSize /=32;
		realSize++;
	}
	else{
		realSize /=32;
	}
	this->decompressedSize = realSize;
	if(cudaSuccess != cudaMalloc((void**)&(this->result_gpu), sizeof(int) * outputSize)){
		std::cout << "Decomp: Could not allocate space for results array" << std::endl;
	}

	no_sorting::decompressWords<<<this->blockCount,blockDim>>>(this->data_gpu, this->result_gpu, this->orderingArray_gpu, this->counts_gpu, this->blockCount, this->compressedSize);
	cudaFree(this->data_gpu);
	cudaFree(this->orderingArray_gpu);
	cudaFree(this->counts_gpu);

	this->blockCount = outputSize / 1024;

	if(this->compressedSize % 1024 > 0){
		this->blockCount++;
	}

	cudaError_t res = cudaMalloc((void**)&(this->finalOutput_gpu), sizeof(int)*outputSize);
	if(cudaSuccess != res){
		std::cout << "Error" << std::endl;
		std::cout << cudaGetErrorName(res) << std::endl;
		std::cout << "Decomp: Could not allocate space for final output array" << std::endl;

}
	cudaMalloc((void**)&this->finalOutput_gpu, sizeof(int)*this->decompressedSize);
	no_sorting::mergeWords<<<this->blockCount,blockDim>>>(this->result_gpu, this->finalOutput_gpu, this->decompressedSize);
	cudaFree(this->result_gpu);
}

template<class T>
T* NoSortPackage<T>::getOrderingArray(){
	return this->orderingArray;
}

template<class T>
T NoSortPackage<T>::getOrderingLength(){
	return this->blockCount;
}

template<class T>
T* NoSortPackage<T>::getBlockSizes(){
	return this->blockSizes;
}

template class NoSortPackage<unsigned long long int>;
template class NoSortPackage<unsigned int>;
