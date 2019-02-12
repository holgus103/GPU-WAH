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

public:

	CompressedPackage();
	virtual ~CompressedPackage();
	bool performAssert();
	void compressData(unsigned int* p_data, unsigned long long int p_size);
};

#endif /* COMPRESSOR_H_ */
