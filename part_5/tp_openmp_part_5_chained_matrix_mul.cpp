// chaining matrix multiplication using OpenMP
// NB: matrices are square identity matrices to simplify checking

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <cmath>

#include "omp.h"

#include "utils.hpp"
#include <iostream>

#define TOL  0.001


int main(int argc, char **argv)
{
    size_t dim = 100; // nb of dimension of square matrices
	size_t nb_matrices = 8; // nb of matrices to chain

    // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-D" ) == 0 )) {
            dim = (size_t) atoi( argv[ ++i ] );
            printf( "  User dim is %ld\n", dim );
        } else if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
			int pow_of_two = atoi( argv[ ++i ] );
			printf( "  User 2^pow_of_two is %ld\n", nb_matrices );
			// NOTE: 1 << n is the same as raising 2 to the power n, or 2^n
			// StackOverflow: https://stackoverflow.com/a/30357743/10798114
			nb_matrices = 1 << pow_of_two;
			printf( "  So nb_matrices is %ld\n", nb_matrices );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
			printf( "  Matrix multiplication Options:\n" );
			printf( "  -D <int>:              Number of dimension of square matrices (by default 1000)\n" );
			printf( "  -N <int>:              Number of matrices to chain (by default 10). Should be even.\n" );
			printf( "  -help (-h):            print this message\n\n" );
			exit( 1 );
        }
    }
	
	// create an array of pointers to matrices
	double **matrices = (double **)MallocOrDie(nb_matrices*sizeof(double *));
      
	// Allocate memory for the matrices.
	for(size_t i = 0; i < nb_matrices; i++) {
		matrices[i] = (double *)MallocOrDie(dim*dim*sizeof(double));
	}

	// Initialize matrices
	for(size_t i = 0; i < nb_matrices; i++) {
		initIdentityMatrixArray(matrices[i], dim, dim);
	}
	printf("Initializing matrices done.");

	// printf("\n First matrix first 10x10 chunk:");
	// for (size_t i = 0; i < 10; i++) {
	// 	printf("\n");
	// 	for (size_t j = 0; j < 10; j++) {
	// 		printf(" %f", matrices[0][j*dim + i]);
	// 	}
	// }
	// printf("\n");

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

	// Chained matrix multiplication
	// top matrices are multplied 2 by 2
	// each result is stored in the even index matrix
	// then the reduction continue until the last matrix[0]
	// which is the result of the multiplication of all matrices
	// log2(nb_matrices) is the number of reduction
	size_t nb_reduction_loops = log2(nb_matrices);
	for (size_t n = 0; n < nb_reduction_loops; n++) { // red level
		// compute number of multiplication in this reduction level
		size_t nb_mult = nb_matrices / (2*(n+1));
		for(size_t m = 0; m < nb_mult; m++) {
			// compute the index of the matrices A and B to multiply
			size_t a = 2*m*(n+1);
			// NOTE: 1 << n is the same as raising 2 to the power n, or 2^n
			// StackOverflow: https://stackoverflow.com/a/30357743/10798114
			size_t b = a + (1 << n); // 2^n

			// allocate memory for the result matrix
			double * C = (double *)MallocOrDie(dim*dim*sizeof(double));

			std::cout << "n=" << n << ", m=" << m << ", a=" << a << ", b=" << b << std::endl;

			// multiply matrices A and B and store the result in C
			for (size_t i = 0; i < dim; i++) {
				for (size_t j = 0; j < dim; j++) {
					double tmp = 0.0;
					for (size_t k = 0; k < dim; k++) {
						double * A = matrices[a];
						double * B = matrices[b];
						double tmpA = A[i*dim + k];
						double tmpB = B[k*dim + j];
						tmp += tmpA * tmpB;
						//tmp += matrices[a][i*dim + k] * matrices[b][k*dim + j];
					}
					C[i*dim + j] = tmp;
				}
			}
			
			// free memory of matrix A and B
			free(matrices[a]);
			free(matrices[b]);

			// store the result in matrix A
			matrices[a] = C;
			
		}
	}

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * (end.tv_sec - begin.tv_sec) +
        1.0e-6 * (end.tv_usec - begin.tv_usec);
                
	printf(" %ld chained matrix multiplications in %f seconds \n", nb_matrices, time);

	// check the result matrices[0] is still identity
	double errsq = 0.0;
	for (size_t i = 0; i < dim; i++) {
		for (size_t j = 0; j < dim; j++) {
			// expected value is 1 if i == j
			// expected value is 0 if i != j
			// expected = (i == j) ? 1.0 : 0.0; === (i == j)
			double err = matrices[0][j*dim + i] - (i == j);
			
			// sum error squares
			errsq += err * err;
		}
	}

	if (errsq > TOL) {
		printf("\n Errors in multiplication: %f", errsq);
	} else {
		printf("\n Hey, it worked");
	}

	// print the result matrix first 10x10 chunk
	printf("\n Result matrix first 10x10 chunk:");
	for (size_t i = 0; i < 10; i++) {
		printf("\n");
		for (size_t j = 0; j < 10; j++) {
			printf(" %f", matrices[0][j*dim + i]);
		}
	}

	// Free up space
	free(matrices[0]); // last matrix is the result
	free(matrices);

	printf("\n all done \n");
}
