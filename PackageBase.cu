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
	//initialize pointers as NULL
	this->data = NULL;
	this->compressedData = NULL;
	this->decompressedData = NULL;

}

PackageBase::~PackageBase() {
	free(this->decompressedData);
	free(this->compressedData);
	// TODO Auto-generated destructor stub
}

void PackageBase::startTimer(){
	cudaEventCreate(&(this->start));
	cudaEventRecord(this->start,0);
}

void PackageBase::getTimerMeasurement(float* pOutVal){
	cudaEventCreate(&(this->stop));
	cudaEventRecord(this->stop,0);
	cudaEventSynchronize(this->stop);
	cudaEventElapsedTime(pOutVal, this->start,this->stop);
}

void PackageBase::c_initializeVariables(unsigned int* in_data, unsigned long long int in_size){

	this->c_compression = 0;
	this->c_transferFromDevice = 0;
	this->c_transferToDevice = 0;

	this->d_compression = 0;
	this->d_transferFromDevice = 0;
	this->d_transferToDevice = 0;
	this->data = in_data;
}


void PackageBase::compressData(unsigned int* in_data, unsigned long long int in_size){
	if(this->compressedData != NULL){
		free(this->compressedData);
		this->compressedData = NULL;
	}
	// template method
	// start measuring time
	// cudaEventCreate(&(this->start));
	// cudaEventRecord(this->start,0);
	this->startTimer();

	this->c_initializeVariables(in_data, in_size);
	this->c_allocateMemory();
	this->c_copyToDevice();

	// get transfer and allocation time
	// cudaEventCreate(&(this->stop));
	// cudaEventRecord(this->stop,0);
	// cudaEventSynchronize(this->stop);
	// cudaEventElapsedTime(&(this->c_transferToDevice), this->start,this->stop);
	this->getTimerMeasurement(&(this->c_transferToDevice));
	// restart time measuring
	// cudaEventCreate(&(this->start));
	// cudaEventRecord(this->start,0);
	this->startTimer();

	this->c_runAlgorithm();

	// get compression time
	// cudaEventCreate(&(this->stop));
	// cudaEventRecord(this->stop,0);
	// cudaEventSynchronize(this->stop);
	// cudaEventElapsedTime(&(this->c_compression), this->start, this->stop);
	this->getTimerMeasurement(&(this->c_compression));
	// restart time measuring
	// cudaEventCreate(&(this->start));
	// cudaEventRecord(this->start,0);
	this->startTimer();

	this->c_copyFromDevice();
	this->c_cleanup();

		// get transfer time
	// cudaEventCreate(&(this->stop));
	// cudaEventRecord(this->stop,0);
	// cudaEventSynchronize(this->stop);
	// cudaEventElapsedTime(&(this->c_transferFromDevice), this->start, this->stop);
	this->getTimerMeasurement(&(this->c_transferFromDevice));
}


void PackageBase::decompressData(){
	if(this->decompressedData != NULL){
		free(this->decompressedData);
		this->decompressedData = NULL;
	}
	// template method
	this->startTimer();
	this->d_initializeVariables();
	this->d_allocateMemory();
	this->d_copyToDevice();
	this->getTimerMeasurement(&(this->d_transferToDevice));
	this->startTimer();
	this->d_runAlgorithm();
	this->getTimerMeasurement(&(this->d_compression));
	this->startTimer();
	this->d_copyFromDevice();
	this->getTimerMeasurement(&(this->d_transferFromDevice));
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
