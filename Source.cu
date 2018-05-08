
#include "compress.h"
#include <stdlib.h>


int main()
{
	int data[] = { 1,2,3,4,5,6,7,8,9,10 };
	int* results = compress(data, 10);
	free(results);
	return 0;
}