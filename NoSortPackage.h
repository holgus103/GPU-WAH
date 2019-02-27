/*
* NoSortPackage.h
*
*  Created on: Feb 13, 2019
*      Author: holgus103
*/
#include "CompressedPackage.h"
#ifndef NOSORTPACKAGE_H_
#define NOSORTPACKAGE_H_

template<class T>
class NoSortPackage : public CompressedPackage<T> {
private:
	T* blockSizes;
	T* orderingArray;
	T* orderingArray_gpu;
    T* sizeCounter_gpu;

public:
	NoSortPackage();
	virtual ~NoSortPackage();
	virtual void c_allocateMemory();
	virtual void c_runAlgorithm();
	virtual void c_cleanup();
	virtual void d_allocateMemory();
	virtual void d_copyToDevice();
	virtual void d_runAlgorithm();
    T* getOrderingArray();
    T getOrderingLength();
    T* getBlockSizes();

};

#endif /* NOSORTPACKAGE_H_ */
