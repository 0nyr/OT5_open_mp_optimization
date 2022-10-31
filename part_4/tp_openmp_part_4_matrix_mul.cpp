/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program. 
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/


#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001

// StackOverflow: https://stackoverflow.com/questions/26831981/should-i-check-if-malloc-was-successful
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

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	int i,j,k;
	double *A, *B, *C, cval, tmp, err, errsq;

    // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
            Ndim = atoi( argv[ ++i ] );
            printf( "  User N is %d\n", Ndim );
        } else if ( ( strcmp( argv[ i ], "-M" ) == 0 )) {
            Mdim = atoi( argv[ ++i ] );
            printf( "  User M is %d\n", Mdim );
        } else if ( ( strcmp( argv[ i ], "-P" ) == 0 )) {
            Pdim = atoi( argv[ ++i ] );
            printf( "  User P is %d\n", Pdim );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
            printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
            printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	// Allocate memory for the matrices.
	A = (double *)MallocOrDie(Ndim*Pdim*sizeof(double));
	B = (double *)MallocOrDie(Pdim*Mdim*sizeof(double));
	C = (double *)MallocOrDie(Ndim*Mdim*sizeof(double));

	// check for successful memory allocation
	if (A == NULL || B == NULL || C == NULL) {
		printf("Error: Can't allocate memory for matrices. Aborting...\n");
		exit(1);
	}

	/* Initialize matrices */

	for (i=0; i<Ndim; i++)
		for (j=0; j<Pdim; j++)
			*(A+(i*Ndim+j)) = AVAL;

	for (i=0; i<Pdim; i++)
		for (j=0; j<Mdim; j++)
			*(B+(i*Pdim+j)) = BVAL;

	for (i=0; i<Ndim; i++)
		for (j=0; j<Mdim; j++)
			*(C+(i*Ndim+j)) = 0.0;

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
    for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			tmp = 0.0;
			for(k=0; k<Pdim; k++){
				/* C(i,j) = sum(over k) A(i,k) * B(k,j) */
				tmp += *(A+(i*Ndim+k)) *  *(B+(k*Pdim+j));
			}
			*(C+(i*Ndim+j)) = tmp;
		}
	}
	/* Check the answer */


    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
        1.0e-6 * (end.tv_usec - begin.tv_usec);
                
	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

	double dN, dM, dP, mflops;
	dN = (double)Ndim;
	dM = (double)Mdim;
	dP = (double)Pdim;
    mflops = 2.0 * dN * dM * dP/(1000000.0* time); // correction 
 
	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);

	cval = Pdim * AVAL * BVAL;
	errsq = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = *(C+i*Ndim+j) - cval;
		    errsq += err * err;
		}
	}

	if (errsq > TOL) 
		printf("\n Errors in multiplication: %f",errsq);
	else
		printf("\n Hey, it worked");

	// Free up space
	free(A);
	free(B);
	free(C);

	printf("\n all done \n");
}
