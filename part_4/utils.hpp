#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>


/**
 * @brief Safe malloc that checks for memory allocation errors.
 * Aborts the program if the allocation fails.
 * StackOverflow: https://stackoverflow.com/questions/26831981/should-i-check-if-malloc-was-successful
 * 
 * @param MemSize 
 * @return void*  
 */
static inline void *MallocOrDie(size_t MemSize)
{
    void *AllocMem = malloc(MemSize);
    /* Some implementations return null on a 0 length alloc,
     * we may as well allow this as it increases compatibility
     * with very few side effects */
    if(!AllocMem && MemSize) // If AllocMem is NULL and MemSize is not 0
	{
        printf("Could not allocate memory!");
        abort();
    }
    return AllocMem;
}


/**
 * @brief Initialize a 2D array of type T with the given value.
 * 
 * @tparam colunms
 * @tparam rows
 * @tparam T
 * @param initValue
 */
template <typename T>
void initArray(T *& array, size_t rows, size_t colunms, T initValue)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < colunms; j++)
        {
            array[i*colunms + j] = initValue;
        }
    }
}
