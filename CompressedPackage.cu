#include "CompressedPackage.h"
#include "tests.h"
#include "cuda_runtime_api.h"
#include "kernels.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include "timeMeasuring.h"
#include "decompress.h"
#include <thrust/scan.h>

template<class T>
CompressedPackage<T>::CompressedPackage()
{

};

template<class T>
CompressedPackage<T>::~CompressedPackage()
{

};

template<class T>
void CompressedPackage<T>::c_initializeVariables(unsigned int* in_data){
	PackageBase::c_initializeVariables(in_data);

	// -- Variable initialization --
	// calculate the number of blocks necessary
	this->blockCount = this->size / (31*32);

	// if not divisible, add another additional block
	if(this->size % (31*32)> 0){
		this->blockCount++;
	}

	// calculate max output size (one extra bit for every 31 bits)
	this->maxExpectedSize = 8*sizeof(int)*this->size;
	if(this->maxExpectedSize % 31 > 0){
		this->maxExpectedSize /= 31;
		this->maxExpectedSize++;
	}
	else{
		this->maxExpectedSize /= 31;
	}
}

template<class T>
void CompressedPackage<T>::c_allocateMemory(){
	// -- Memory allocation --

		// allocate memory on the device
		if(cudaSuccess != cudaMalloc((void**)&(this->data_gpu), this->size * sizeof(int))){
			std::cout << "Could not allocate space for the data" << std::endl;
			return;
		}
		if(cudaSuccess != cudaMalloc((void**)&(this->compressed_gpu), this->maxExpectedSize * sizeof(int))){
			std::cout << "Could not allocate space for the compressed output" << std::endl;
			cudaFree(this->data_gpu);
			return;
		}
		if(cudaSuccess != cudaMalloc((void**)&(this->blockCounts_gpu), this->blockCount* sizeof(T))){
			std::cout << "Could not allocate space for the block sizes" << std::endl;
			cudaFree(this->data_gpu);
			cudaFree(this->compressed_gpu);
			return;
		}
}

template<class T>
void CompressedPackage<T>::c_copyToDevice(){
	// -- Data transfer --

		// copy input
		if(cudaSuccess != cudaMemcpy(this->data_gpu, this->data, this->size*sizeof(int), cudaMemcpyHostToDevice)){
			std::cout << "Could not copy input" << std::endl;
			cudaFree(this->data_gpu);
			cudaFree(this->compressed_gpu);
			cudaFree(this->blockCounts_gpu);
			return;
		}
}

template<class T>
void CompressedPackage<T>::c_runAlgorithm(){
	// -- Data compression --

			// initialize block dimensions
			dim3 blockSize = dim3(32, 32, 1);

			// call compression kernel, merges words within a block
			regular_kernels::gpu_compressData<T><<<this->blockCount, blockSize>>>(this->data_gpu, this->compressed_gpu, this->blockCounts_gpu, this->size);

			// remove unnecessary data
			cudaFree((void*)this->data_gpu);
			thrust::device_ptr<T> blockCountsPtr(this->blockCounts_gpu);


			T lastWordNumber;

			// get the size of the last block
			if(cudaSuccess != cudaMemcpy(&lastWordNumber, this->blockCounts_gpu + (this->blockCount - 1), sizeof(T), cudaMemcpyDeviceToHost)){
				std::cout << "Could not copy last block count" << std::endl;
				cudaFree(this->compressed_gpu);
				cudaFree(this->blockCounts_gpu);
				return;
			}


			// perform a scan to get offsets
			thrust::exclusive_scan(blockCountsPtr, blockCountsPtr + this->blockCount, blockCountsPtr);
			T  lastBlockOffset;

			// get the offset of the last block
			if(cudaSuccess != cudaMemcpy(&lastBlockOffset, this->blockCounts_gpu + (this->blockCount - 1), sizeof(T), cudaMemcpyDeviceToHost)){
				std::cout << "Could not copy last block offset" << std::endl;
				cudaFree(this->compressed_gpu);
				cudaFree(this->blockCounts_gpu);
				return;
			}

			T outputSize = lastBlockOffset + lastWordNumber;
			// save compressed size
			SAFE_ASSIGN(&(this->compressedSize), outputSize)
			if(cudaSuccess != cudaMalloc((void**)&(this->finalOutput_gpu), sizeof(int) * outputSize)){
				std::cout << "Could not allocate final Output" << std::endl;
				cudaFree(this->compressed_gpu);
				cudaFree(this->blockCounts_gpu);
				return;
			}
			// call merge kernel
			regular_kernels::gpu_moveData<T><<<this->blockCount, blockSize>>>(this->compressed_gpu, this->finalOutput_gpu, this->blockCounts_gpu);
}

template<class T>
void CompressedPackage<T>::c_copyFromDevice(){
	// -- Move decompressed data from device to host --

		// allocate memory for results
		this->compressedData = (unsigned int*)malloc(sizeof(int) * this->compressedSize);

		// copy compressed data
		if(cudaSuccess != cudaMemcpy((void*)this->compressedData, (void*)this->finalOutput_gpu, this->compressedSize * sizeof(int), cudaMemcpyDeviceToHost)){
			std::cout << "Could not copy final output" << std::endl;
		}
}

template<class T>
void CompressedPackage<T>::c_cleanup(){
		// -- Cleanup --

		// free gpu memory
		cudaFree((void*)this->compressed_gpu);
		cudaFree((void*)this->blockCounts_gpu);
		cudaFree((void*)this->finalOutput_gpu);
}

template<class T>
bool CompressedPackage<T>::performAssert(){
	if(this->decompressedSize == this->size){
		ASSERT(this->decompressedData, this->data, this->size);
	}
return false;
};

template<class T>
void CompressedPackage<T>::d_initializeVariables(){
	// start measuring time
	CREATE_TIMER
	START_TIMER

	this->blockCount = this->compressedSize / 1024;

	if(this->compressedSize % 1024 > 0){
		this->blockCount++;
	}
}

template<class T>
void CompressedPackage<T>::d_allocateMemory(){
	//	-- Memory allocation --

	cudaMalloc((void**)&(this->data_gpu), sizeof(int)*this->compressedSize);
	cudaMalloc((void**)&(this->counts_gpu), sizeof(T)*this->compressedSize);
}

template<class T>
void CompressedPackage<T>::d_copyToDevice(){
	// -- Data transfer --
	cudaMemcpy(this->data_gpu, this->compressedData, sizeof(int)*this->compressedSize, cudaMemcpyHostToDevice);

	STOP_TIMER
	GET_RESULT(this->d_transferToDevice)
	START_TIMER
}

template<class T>
void CompressedPackage<T>::d_runAlgorithm(){
	// initialize block dimensions
	dim3 blockDim(32, 32);
	// -- Decompress data --

	// get blocked sizes of all words in the compressed file
	regular_kernels::gpu_getCounts<T><<<this->blockCount,blockDim>>>(this->data_gpu, this->counts_gpu, this->compressedSize);

	// size of the last block
	T lastBlockSize;

	// copy the size of the last block
	cudaMemcpy(&lastBlockSize, this->counts_gpu  + (this->compressedSize - 1), sizeof(T), cudaMemcpyDeviceToHost);

	thrust::device_ptr<T> countsPtr(this->counts_gpu);
	// scan block sizes and save offsets
	thrust::exclusive_scan(countsPtr, countsPtr + this->compressedSize, countsPtr);
	// offset of the last block
	T lastOffset;
	// copy the offset of the last block
	cudaMemcpy(&lastOffset, counts_gpu + (this->compressedSize - 1), sizeof(T), cudaMemcpyDeviceToHost);
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
	this->decompressedSize = realSize;
//	free(counts_cpu);

	// allocate memory for the result
	cudaMalloc((void**)&(this->result_gpu), sizeof(int) * outputSize);

	// decompress the words
	regular_kernels::gpu_decompressWords<T><<<this->blockCount,blockDim>>>(this->data_gpu, this->counts_gpu, this->result_gpu, this->compressedSize);

	// free data and block offsets
	cudaFree(this->data_gpu);
	cudaFree(this->counts_gpu);

	// calculate block count
	int blockCount = outputSize / 1024;
	if(this->compressedSize % 1024 > 0){
		blockCount++;
	}

	// allocate memory for final output data
	cudaMalloc((void**)&(this->finalOutput_gpu), sizeof(int)*outputSize);
	// convert from 31 bit words to integers
	regular_kernels::gpu_mergeWords<T><<<blockCount,blockDim>>>(this->result_gpu, this->finalOutput_gpu, outputSize);
}

template<class T>
void CompressedPackage<T>::d_copyFromDevice(){
		// free incomplete array
		cudaFree(result_gpu);

		STOP_TIMER
		GET_RESULT(this->d_compression)
		START_TIMER
	
		// allocate host array and copy final result
		this->decompressedData = (unsigned int*)malloc(sizeof(int) * this->decompressedSize);
		cudaMemcpy(this->decompressedData, this->finalOutput_gpu, sizeof(int) * this->decompressedSize, cudaMemcpyDeviceToHost);
}

template<class T>
void CompressedPackage<T>::d_cleanup(){

	// free device result array
	cudaFree(this->finalOutput_gpu);
}

template<class T>
void CompressedPackage<T>::d_getStats(){
	STOP_TIMER
	GET_RESULT(this->d_transferFromDevice)
}


template class CompressedPackage<unsigned long long int>;
template class CompressedPackage<unsigned int>;

//
