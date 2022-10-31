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
**           Modified and extended by Jonathan Rouzaud-Cornabas, Oct 202
** 			 Corrected (segfaults) by 0nyr, Oct 2022
*/


#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "omp.h"

#include "utils.hpp"

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001


int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	float *A, *B, *C, cval, tmp, err, errsq;

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
	A = (float *)MallocOrDie(Ndim*Pdim*sizeof(float));
	B = (float *)MallocOrDie(Pdim*Mdim*sizeof(float));
	C = (float *)MallocOrDie(Ndim*Mdim*sizeof(float));

	// Initialize matrices
	initArray(A, Ndim, Pdim, (float) AVAL);
	initArray(B, Pdim, Mdim, (float) BVAL);
	initArray(C, Ndim, Mdim, (float) 0.0);
	printf("Initializing matrices done.");

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
	#pragma omp parallel for firstprivate(Ndim) firstprivate(A) firstprivate(B) firstprivate(C)
    for (int i = 0; i<Ndim; i++) {
		#pragma omp parallel for firstprivate(Mdim) firstprivate(Pdim) private(tmp)
		for (int j = 0; j<Mdim; j++) {
			tmp = 0.0;
			#pragma omp parallel shared(tmp)
      		#pragma omp parallel for simd reduction(+: tmp)
			for(int k = 0; k<Pdim; k++) {
				// C(i,j) = sum(over k) A(i,k) * B(k,j)
				tmp += A[i*Pdim + k] * B[k*Mdim + j];
			}
			C[i*Mdim + j] = tmp; // C(i,j) = tmp
		}
	}

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

	// Check result
	cval = Pdim * AVAL * BVAL;
	errsq = 0.0;
	for (int i = 0; i < Ndim; i++) {
		for (int j = 0; j < Mdim; j++) {
			err = C[i*Mdim+j] - cval;
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
