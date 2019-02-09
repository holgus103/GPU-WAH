#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "device_launch_parameters.h"
#include "kernels.h"
#include "tests.h"
#include <stdio.h>
#include "const.h"

// some defines to avoid repetition
#define IF_LAST if(id == (warpSize - 1))

#define WARP_OPERATION_DOWN(OP, NAME) \
__inline__ __device__ int NAME(int val){ \
	for (int mask = WARP_SIZE/2; mask > 0; mask /= 2)\
		val OP __shfl_xor(val, mask);\
	return val; \
}\

WARP_OPERATION_DOWN(|=, orWithinWarp);

__inline__ __device__ int localScan(int val, int id){
	for(int i = 1; i < 32; i<<=1){
		int ret = __shfl_up(val, i);
		val += id >= i ? ret : 0;
	}
	return val;
}

__inline__ __device__ void markEndWordTypes(int w, int* end, int id){
	IF_LAST{
		end[threadIdx.y] = w;
	}
}

__inline__ __device__ void writeEndingSize(int id, int* lengths, int size){
	IF_LAST{
		lengths[threadIdx.y] = size;
	}
}

/*
 * Device function performing compression
 *
 * Parameters:
 * data - device pointer to data to be compressed
 * output - device pointer to the output array
 * blockCounts - device pointer to an array with block sizes
 * dataSize - input data size in integers
 */
template<class T>
__global__ void compressData(unsigned int* data, unsigned int* output, T* blockCounts, int dataSize) {
	// count of words for every warp
	__shared__ int counts[32];
	// length of the last word in a warp
	__shared__ int endLengths[32];
	// type of the last word in a warp
	__shared__ int endings[32];
	// type of the first word in a warp
	__shared__ int beginnings[32];
	// array indicating whether the last thread of a warp has been merged
	__shared__ bool merged[32];


// -- Prepare initial variables and read data --

	// get thread id
	int id = threadIdx.x;
	int id_global = blockIdx.x * (31*32) + threadIdx.y *31 + id;
	unsigned int word = 0;
	if(id_global > dataSize) return;
	// retrieve word, only first 31 threads
	if (id < WARP_SIZE - 1) {
		word = data[id_global];
	}
	// divide words into 31bit parts 
	// gets 31 - id bits from one lane above
	// and id bits from own lane
	//word = (__shfl_down(word, 1) & (ONES31 >> id)) << id | (word & TOP31ONES) >> (32 - id);
	word = ONES31 & ((__shfl_up(word, 1) >> (32 - id)) | (word << id));


// -- Recognize and mark words --

	// word info variables
	int ones = 0;
	int zeros = 0;
	int literals = 0;
	int type;

	// mark word types for warp
	// detect words with zeros and words with ones
	// is a zero fill word
	if (word == ZEROS) {
		zeros |= 1 << id;
		type = WORD_ZEROS;
		// last word in a warp marks word as zeros
		markEndWordTypes(WORD_ZEROS, endings, id);
	}

	// is a one fill word
	else if (word == ONES31) {
		ones |= 1 << id;
		type = WORD_ONES;
		// last word in a warp marks word as ones
		markEndWordTypes(WORD_ONES, endings, id);
	}
	else
	{
		type = WORD_LITERAL;
		// last word in a warp marks word as literal
		markEndWordTypes(WORD_LITERAL, endings, id);
	}

	// exchange word information within the warp
	zeros = orWithinWarp(zeros);
	ones = orWithinWarp(ones);
	literals = ~(zeros | ones);

// send complete information to other threads
//	if (id == WARP_LEADER) {
//		zeros = __shfl(zeros, 0);
//		ones = __shfl(ones, 0);
//		literals = __shfl(literals, 0);
//	}

	int n = 0x3 << id;
	int flags = BIT31;
	bool idle = true;

	// if is not last
	if (id < 31) {
		int res = 1 << id;
		if (((n & zeros) == res || (n & ones) == res || (literals & (1 << id)) > 0)) {
			// detect endings of sequences of words of the same type and mark them
			flags |= 1 << id;
			idle = false;
		}
	}
	else{
		idle = false;
	}

// -- Calculate block size --

	// exchange endings 
	flags = orWithinWarp(flags);
	int blockSize = 1;
	// index within warp
	int index = __popc(((1<<id) - 1) & flags);
	// if first word in block, write beginning
	if(index == 0){
		beginnings[threadIdx.y] = type;
	}

	// calculate the number of words within a compressed word
	if (!idle) {
		for (int i = id-1; i >= 0; i--) {
			if ((flags & (1 << i)) > 0) {
				break;
			}
			blockSize++;
		}
		if (word == ONES31) {
//			word = BIT3130;
			writeEndingSize(id, endLengths, blockSize);
		}
		else if (word == ZEROS) {
//			word = BIT31;
			writeEndingSize(id, endLengths, blockSize);
		}
		else{
			writeEndingSize(id, endLengths, 0);
		}
	}

	// last thread calculates the number of words and writes it to the shared array
	if(id == WARP_LEADER){
		counts[threadIdx.y] = __popc(flags);
	}

	// sync all threads within block
	__syncthreads();

// -- Merging warps --

	// the first warp scans the array and gets total block word size
	// then calculates offset
	int mergeShift = 0;
	if(threadIdx.y == BLOCK_LEADER){
		merged[id] = false;
		int count = counts[id];
		// used to not check the same condition twice
		bool satisfiedMergingConditions = false;
		// only execute if the current word will not be merged into another one
		if((id == warpSize - 1) || (endings[id] != beginnings[id+1]) || endings[id] == WORD_LITERAL || counts[id] > 1){
			int i = 1;
			satisfiedMergingConditions = true;
			int bonus = 0;
			// calculate merge shifts
			while(true){
				// check for warps with length 1 and the same word as our beginning
				if(i < id && counts[id - i] == 1 && beginnings[id] == endings[id-i] && beginnings[id] != WORD_LITERAL){
					mergeShift++;
					merged[id - i] = true;
					bonus += endLengths[id - i];
					i++;

				}
				// check for warps that can be partially merged - with the same ending as our beginning
				else if(i <= id && beginnings[id] == endings[id - i] && beginnings[id] != WORD_LITERAL){
					mergeShift++;
					merged[id - i] = true;
					bonus += endLengths[id - i];
					i++;
					break;
				}
				else break;
			}
			endLengths[id] = bonus;
		}
		if(!satisfiedMergingConditions){
			endLengths[id] = 0;
		}
			// let every thread get the shift for its warp
			mergeShift = localScan(mergeShift, id);
			int blockOffset = localScan(count, id);
			// get the offset for the current warp within the block and store it in counts
			counts[id] = blockOffset - count - mergeShift;
	}

	__syncthreads();

	IF_LAST{
		idle = merged[threadIdx.y];
	}

// -- Writing final output for the block --

	// get global offset for warp and warp offset
	if(!idle){
		// first word in a warp gets a bonus
		int bonus = index == 0 ? endLengths[threadIdx.y] : 0;
		index += counts[threadIdx.y];
		if (word == ONES31) {
			word = BIT3130 | (blockSize + bonus);
		}
		else if (word == ZEROS) {
			word = BIT31 | (blockSize + bonus);
		}

		// if it's the last thread in block - either processing last word or the last thread of the last warp
		if((id == (warpSize - 1) && threadIdx.y == (blockDim.y - 1)) || id_global == (dataSize - 1)){
				blockCounts[blockIdx.x] = index + 1;

		}
		output[index + (blockDim.x * blockDim.y) * blockIdx.x] = word;


	}


}

template __global__ void compressData<unsigned long long int>(unsigned int* data, unsigned int* output, unsigned long long int* blockCounts, int dataSize);
template __global__ void compressData<unsigned int>(unsigned int* data, unsigned int* output, unsigned int* blockCounts, int dataSize);


/*
 * Device function moving data from different compressed blocks and removing gaps
 *
 * Parameters:
 * initialOutput - device pointer to the compressed data with gaps
 * finalOutput - device pointer to the output array
 * blockCounts - device pointer to an array with block sizes
 */
template<class T>
__global__ void moveData(unsigned int* initialOutput, unsigned int* finalOutput, T* blockCounts){
	int globalId = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int word = initialOutput[globalId];
	if(word == 0) return;
	unsigned int blockOffset = blockCounts[blockIdx.x];
	int blockId = threadIdx.x + threadIdx.y * blockDim.x;
	finalOutput[blockOffset + blockId] = word;
}

template __global__ void moveData<unsigned long long int>(unsigned int* initialOutput, unsigned int* finalOutput, unsigned long long int* blockCounts);
template __global__ void moveData<unsigned int>(unsigned int* initialOutput, unsigned int* finalOutput, unsigned int* blockCounts);


/*
 * Device function moving data from different compressed blocks and removing gaps
 *
 * Parameters:
 * data_gpu - device pointer to the compressed data
 * counts_gpu - device pointer to an array storing sizes of blocks
 * dataSize - input data size in integers
 */
template<class T>
__global__ void getCounts(unsigned int* data_gpu, T* counts_gpu, T dataSize){
	// get global id
	int globalId = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
	// is within the data range
	if(globalId < dataSize){
		// get word
		unsigned int word = data_gpu[globalId];
		if((BIT31 & word) > 0){
			// if filler word - get count
				int count = word & (BIT30 - 1);
				counts_gpu[globalId] = count;
		}
		else{
			counts_gpu[globalId] = 1;
		}

	}

}

template __global__ void getCounts<unsigned long long int>(unsigned int* data_gpu, unsigned long long int* counts_gpu, unsigned long long int dataSize);
template __global__ void getCounts<unsigned int>(unsigned int* data_gpu, unsigned int* counts_gpu, unsigned int dataSize);


/*
 * Device function performing decompression
 *
 * Parameters:
 * data_gpu - device pointer to the compressed data
 * counts_gpu - device pointer to an array storing sizes of blocks
 * result_gpu - device pointer to the output array
 * dataSize - input data size in integers
 */
template<class T>
__global__ void decompressWords(unsigned int* data_gpu, T* counts_gpu, unsigned int* result_gpu, T dataSize){
	// get global id
	unsigned long long int globalId = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
	// out of range
	if(globalId >= dataSize) return;
	// read word
	unsigned int word = data_gpu[globalId];
	// read offset for block
	unsigned long long int offset = counts_gpu[globalId];
	//printf("id : %d offset: %d \n", globalId, offset);
	// is filler word
	if((BIT31 & word) > 0){

		int count = word & (BIT30 - 1);
		unsigned int filler;
		// assign correct filler word
		if((BIT3130 & word) == BIT3130){
			// is ones
			filler = ONES31;
		}
		else{
			// zeros
			filler = 0;
		}
		// fill array
		for(int i = 0; i < count; i++){
			result_gpu[offset + i] = filler;
		}

	}
	// is literal word
	else{
		result_gpu[offset] = word;
	}
//	printf("%d", offset);

}

template __global__ void decompressWords<unsigned long long int>(unsigned int* data_gpu, unsigned long long int* counts_gpu, unsigned int* result_gpu, unsigned long long int dataSize);
template __global__ void decompressWords<unsigned int>(unsigned int* data_gpu, unsigned int* counts_gpu, unsigned int* result_gpu, unsigned int dataSize);

/*
 * Device function converting 32 31-bit words into 31 32-bit ones
 *
 * Parameters:
 * result_gpu - device pointer to the decompressed data
 * finalOutput_gpu - device pointer to the final output array
 * dataSize - input data size in integers
 */
template<class T>
__global__ void mergeWords(unsigned int* result_gpu, unsigned int* finalOutput_gpu, T dataSize){
	// get global id
	int globalId = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
	int id = threadIdx.x;
	if(globalId >= dataSize) return;
	unsigned int word = result_gpu[globalId];
	word = (__shfl_down(word, 1) << (warpSize - id - 1)) | (word >> id);
	// first 31 threads save a word each
	if(id < warpSize - 1){
		finalOutput_gpu[blockIdx.x * 31*32 + threadIdx.y * 31 + id] = word;
//		if(blockIdx.x * 31*32 + threadIdx.y * 31 + id == 31){
//			printf("thread id %d", globalId);
//		}
	}


}

template __global__ void mergeWords<unsigned long long int>(unsigned int* result_gpu, unsigned int* finalOutput_gpu, unsigned long long int dataSize);
template __global__ void mergeWords<unsigned int>(unsigned int* result_gpu, unsigned int* finalOutput_gpu, unsigned int dataSize);
