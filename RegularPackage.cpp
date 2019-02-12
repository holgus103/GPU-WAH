/*
 * RegularPackage.cpp
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */

#include "RegularPackage.h"
#include "compress.h"
#include "decompress.h"
#include "timeMeasuring.h"
#include <stdlib.h>

template<class T>
RegularPackage<T>::RegularPackage() {
	// TODO Auto-generated constructor stub

}

template<class T>
RegularPackage<T>::~RegularPackage() {
	// TODO Auto-generated destructor stub
}

template<class T>
void RegularPackage<T>::compressData(unsigned int* p_data, unsigned long long int p_size){
	if(this->compressedData != NULL){
		free(this->compressedData);
		this->compressedData = NULL;
	}

	CompressedPackage<T>::compressData(p_data, p_size);
	this->compressedData = compress<T>(
			this->data,
			this->size,
			&(this->compressedSize),
			&(this->c_transferToDevice),
			&(this->c_compression),
			&(this->c_transferFromDevice)
	);

}

template<class T>
unsigned int* RegularPackage<T>::decompressData(){

	if(this->decompressedData != NULL){
		free(this->decompressedData);
		this->decompressedData = NULL;
	}

	this->decompressedData = decompress<T>(
			this->compressedData,
			this->compressedSize,
			&(this->decompressedSize),
			&(this->d_transferToDevice),
			&(this->d_compression),
			&(this->d_transferFromDevice)
	);
}

template class RegularPackage<unsigned int>;
template class RegularPackage<unsigned long long int>;
