/*
 * PackageBase.h
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */


#ifndef PACKAGEBASE_H_
#define PACKAGEBASE_H_

#include <vector_types.h>

class PackageBase {
protected:
	float c_transferToDevice;
	float c_transferFromDevice;
	float c_compression;
	float d_transferToDevice;
	float d_transferFromDevice;
	float d_compression;
	cudaEvent_t start, stop;
	unsigned int* data;
	unsigned int* compressedData;
	unsigned int* decompressedData;
	virtual void c_initializeVariables(unsigned int* in_data);
	virtual void c_allocateMemory() = 0;
	virtual void c_copyToDevice() = 0;
	virtual void c_runAlgorithm() = 0;
	virtual void c_copyFromDevice() = 0;
	virtual void c_cleanup() = 0;
	virtual void c_getStats() = 0;
	virtual void d_initializeVariables() = 0;
	virtual void d_allocateMemory() = 0;
	virtual void d_copyToDevice() = 0;
	virtual void d_runAlgorithm() = 0;
	virtual void d_copyFromDevice() = 0;
	virtual void d_cleanup() = 0;
	virtual void d_getStats() = 0;
public:

	struct Times{
		float c_transferToDevice;
		float c_transferFromDevice;
		float c_compression;
		float d_transferToDevice;
		float d_transferFromDevice;
		float d_compression;
	};
	PackageBase();
	virtual ~PackageBase();
	virtual bool performAssert() = 0;
	unsigned int* getCompressed();
	unsigned int* getDecompressed();
	Times getTimes();
	virtual void compressData(unsigned int* data, unsigned long long int size);
	virtual void decompressData();
};

#endif /* PACKAGEBASE_H_ */
