/*
 * RegularPackage.h
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */
#include "CompressedPackage.h"

#ifndef REGULARPACKAGE_H_
#define REGULARPACKAGE_H_

template<class T>
class RegularPackage : public CompressedPackage<T> {
public:
	RegularPackage();
	virtual ~RegularPackage();
	virtual void compressData(unsigned int* p_data, unsigned long long int p_size);
	virtual unsigned int* decompressData();


};

#endif /* REGULARPACKAGE_H_ */
