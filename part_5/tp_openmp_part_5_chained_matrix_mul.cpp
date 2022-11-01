// chaining matrix multiplication using OpenMP
// NB: matrices are square identity matrices to simplify checking

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "omp.h"

#include "utils.hpp"

#define TOL  0.001


int main(int argc, char **argv)
{
    int dim = 1000; // nb of dimension of square matrices
	int nb_matrices = 10; // nb of matrices to chain
	double *A, *B, *C, cval, tmp, err, errsq;

    // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-D" ) == 0 )) {
            dim = atoi( argv[ ++i ] );
            printf( "  User dim is %d\n", dim );
        } else if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
			nb_matrices = atoi( argv[ ++i ] );
			printf( "  User nb_matrices is %d\n", nb_matrices );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
			printf( "  Matrix multiplication Options:\n" );
			printf( "  -D <int>:              Number of dimension of square matrices (by default 1000)\n" );
			printf( "  -N <int>:              Number of matrices to chain (by default 10)\n" );
			printf( "  -help (-h):            print this message\n\n" );
			exit( 1 ); 
        }
    }
	
	// create an array of pointers to matrices
	double **matrices = (double **)MallocOrDie(nb_matrices*sizeof(double *));
      
	// Allocate memory for the matrices.
	for(int i = 0; i < nb_matrices; i++) {
		matrices[i] = (double *)MallocOrDie(dim*dim*sizeof(double));
	}

	// Initialize matrices
	for(int i = 0; i < nb_matrices; i++) {
		initIdentityMatrixArray(matrices[i], dim, dim);
	}
	printf("Initializing matrices done.");

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    for (int i = 0; i<Ndim; i++) {
		for (int j = 0; j<Mdim; j++) {
			tmp = 0.0;
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
