/*
 * CompresedPackage.h
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */
#include "PackageBase.h"

#ifndef COMPRESSOR_H_
#define COMPRESSOR_H_

template<class T>
class CompressedPackage : public PackageBase{
protected:
	T compressedSize;
	T decompressedSize;
	T size;
	T blockCount;
	// device pointers used during compression
	unsigned int* data_gpu;
	unsigned int* compressed_gpu;
	unsigned long long int maxExpectedSize;
	T* blockCounts_gpu;
	// device pointers using decompression
	unsigned int* result_gpu;
	unsigned int* finalOutput_gpu;
	T* counts_gpu;

public:

	CompressedPackage();
	virtual ~CompressedPackage();
	virtual void c_initializeVariables(unsigned int* in_data);
	virtual void c_allocateMemory();
	virtual void c_copyToDevice();
	virtual void c_runAlgorithm();
	virtual void c_copyFromDevice();
	virtual void c_cleanup();
	virtual void c_getStats();
	virtual void d_initializeVariables();
	virtual void d_allocateMemory();
	virtual void d_copyToDevice();
	virtual void d_runAlgorithm();
	virtual void d_copyFromDevice();
	virtual void d_cleanup();
	virtual void d_getStats();
	bool performAssert();
	//void compressData(unsigned int* p_data, unsigned long long int p_size);
};

#endif /* COMPRESSOR_H_ */
