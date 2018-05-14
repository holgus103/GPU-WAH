
#include "compress.h"
#include <stdlib.h>
#include "tests.h"

int main()
{
//	warpCompressionTest
	blockCompressionTest();
	blockMergeTest();
	return 0;
}
