/*
 * RegularPackage.h
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */
#include "CompressedPackage.h"
#include <thrust/device_ptr.h>

#ifndef REGULARPACKAGE_H_
#define REGULARPACKAGE_H_

template<class T>
class RegularPackage : public CompressedPackage<T> {
protected:
	thrust::device_ptr<T> blockCountsPtr;
public:
	RegularPackage();
	virtual ~RegularPackage();
	virtual void compressData(unsigned int* p_data, unsigned long long int p_size);
	virtual void decompressData();


};

#endif /* REGULARPACKAGE_H_ */
