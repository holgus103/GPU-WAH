
#include "compress.h"
#include <stdlib.h>
#include "tests.h"

int main()
{

#ifdef WORD_DIVISION_TEST
	divideIntoWordsTest();
#endif // WORD_DIVISION_TEST
#ifdef EXTENSION_TEST
	extendDataTest();
#endif
	return 0;
}
