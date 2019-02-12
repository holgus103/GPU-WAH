/*
 * PackageBase.h
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */

#ifndef PACKAGEBASE_H_
#define PACKAGEBASE_H_

class PackageBase {
protected:
	float c_transferToDevice;
	float c_transferFromDevice;
	float c_compression;
	float d_transferToDevice;
	float d_transferFromDevice;
	float d_compression;
	unsigned int* data;
	unsigned int* compressedData;
	unsigned int* decompressedData;
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
	virtual unsigned int* decompressData() = 0;
};

#endif /* PACKAGEBASE_H_ */
