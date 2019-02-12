/*
 * PackageBase.cpp
 *
 *  Created on: Feb 10, 2019
 *      Author: holgus103
 */

#include "PackageBase.h"
#include "tests.h"
#include <stdlib.h>

PackageBase::PackageBase() {
	// TODO Auto-generated constructor stub
	this->data = NULL;
	this->compressedData = NULL;
	this->decompressedData = NULL;

}

PackageBase::~PackageBase() {
	free(this->decompressedData);
	free(this->compressedData);
	// TODO Auto-generated destructor stub
}




void PackageBase::compressData(unsigned int* in_data, unsigned long long int in_size){
	this->c_compression = 0;
	this->c_transferFromDevice = 0;
	this->c_transferToDevice = 0;

	this->d_compression = 0;
	this->d_transferFromDevice = 0;
	this->d_transferToDevice = 0;

	this->data = in_data;
}

unsigned int* PackageBase::getCompressed(){
	return this->compressedData;
}

unsigned int* PackageBase::getDecompressed(){
	return this->decompressedData;
}

PackageBase::Times PackageBase::getTimes(){
	Times t;
	t.c_compression = this->c_compression;
	t.c_transferFromDevice = this->c_transferFromDevice;
	t.c_transferToDevice = this->c_transferToDevice;
	t.d_compression = this->d_compression;
	t.d_transferFromDevice = this->d_transferFromDevice;
	t.d_transferToDevice = this->d_transferToDevice;
	return t;
}
