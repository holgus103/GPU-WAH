#include "CompressedPackage.h"
#include "tests.h"
#include <stdlib.h>

template<class T>
CompressedPackage<T>::CompressedPackage()
{

};

template<class T>
CompressedPackage<T>::~CompressedPackage()
{

};


template<class T>
bool CompressedPackage<T>::performAssert(){
	if(this->decompressedSize == this->size){
		ASSERT(this->decompressedData, this->data, this->size);
	}
	return false;
};

template<class T>
void CompressedPackage<T>::compressData(unsigned int* p_data, unsigned long long int p_size){
	PackageBase::compressData(p_data, p_size);
	this->size = p_size;
}


template class CompressedPackage<unsigned long long int>;
template class CompressedPackage<unsigned int>;

//
