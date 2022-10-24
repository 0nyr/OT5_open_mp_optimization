/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <iostream>

#include <cmath>

using namespace std;

#include <stdlib.h>
#include <malloc.h>
#include <immintrin.h>
#include <memory>

// StackOverflow: https://en.cppreference.com/w/c/memory/aligned_alloc 
template <typename T, std::size_t N = 16>
class AlignmentAllocator {
public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    typedef T * pointer;
    typedef const T * const_pointer;

    typedef T & reference;
    typedef const T & const_reference;

    public:
    inline AlignmentAllocator () throw () { }

    template <typename T2>
    inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }

    inline ~AlignmentAllocator () throw () { }

    inline pointer address (reference r) {
        return &r;
    }

    inline const_pointer address (const_reference r) const {
        return &r;
    }

    inline pointer allocate (size_type n) {
        return (pointer)aligned_alloc(n*sizeof(value_type), N);
    }

    inline void deallocate (pointer p, size_type) {
        free(p);
    }

    inline void construct (pointer p, const value_type & wert) {
        new (p) value_type (wert);
    }

    inline void destroy (pointer p) {
        p->~value_type ();
    }

    inline size_type max_size () const throw () {
        return size_type (-1) / sizeof (value_type);
    }

    template <typename T2>
    struct rebind {
        typedef AlignmentAllocator<T2, N> other;
    };

    bool operator!=(const AlignmentAllocator<T,N>& other) const  {
        return !(*this == other);
    }

    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const AlignmentAllocator<T,N>& other) const {
        return true;
    }
};





void checkSizes(long long &N, long long &M, long long &S, int &nrepeat);

int main( int argc, char* argv[] )
{
  // print file name
  cout << "File: " << __FILE__ << endl;

  long long N = -1;         // number of rows 2^12
  long long M = -1;         // number of columns 2^10
  long long S = -1;         // total size 2^22
  int nrepeat = 100;        // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %lld\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %lld\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %lld\n", S );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, nrepeat );

  // Allocate x,y,A
  // Initialize y vector to 1.
  // Initialize x vector to 1.
  // Initialize A matrix, you can use a 1D index if you want a flat structure (i.e. a 1D array) e.g. j*M+i is the same than [j][i]
  vector<int, AlignmentAllocator<int, 64>>* y = new vector<int, AlignmentAllocator<int, 64>>(N,1);
  vector<int, AlignmentAllocator<int, 64>>* x = new vector<int, AlignmentAllocator<int, 64>>(M,1);
  vector<vector<int, AlignmentAllocator<int, 64>>, AlignmentAllocator<int, 64>>* A = 
    new vector<vector<int, AlignmentAllocator<int, 64>>, AlignmentAllocator<int, 64>>(N, vector<int, AlignmentAllocator<int, 64>>(M,1)); // matrix N*M

  // Timer products.
  struct timeval begin, end;

  gettimeofday( &begin, NULL );

  // WARN: perf evaluation = DON'T PARALLEL !!!
  // #pragma omp parallel for schedule(static)
  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // For each line i
    // Multiply the i lines with the vector x 
    // Sum the results of the previous step into a single variable
    long long result_t1;
    long long result = 0;
    for ( int i = 0; i < N; i++ ) {
      result_t1 = 0;
      result = 0;
      # pragma omp parallel for simd
      for (int j = 0; j < M; j++) {
        result_t1 += (*A)[i][j]*(*x)[j];
      }
      // Multiply the result of the previous step with the i value of vector y
      # pragma omp parallel for simd
      for ( int k = 0; k < N; k++ ) {
        // Sum the results of the previous step into a single variable (result)
        result += (*y)[k]*result_t1;
      }
    }

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %lld x %lld is %lld\n", N, M, result);
    }

    const long long solution = N*M;

    if ( result != solution ) {
      printf( "  Error: result( %lld ) != solution( %lld )\n", result, solution);
    }
  }

  gettimeofday( &end, NULL );

  // Calculate time.
  //double time = timer.seconds();
  double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %lld ) M( %lld ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  delete(A);
  delete(y);
  delete(x);

  return 0;
}

void checkSizes(long long &N, long long &M, long long &S, int &nrepeat) {
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 22 );
    if ( S < N ) S = N;
    if ( S < M ) S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if ( S == -1 ) S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( N == -1 && M == -1 ) {
    if ( S > 1024 ) {
      M = 1024;
    }
    else {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If N is undefined, set it.
  if ( N == -1 ) N = S / M;

  printf( "  Total size S = %lld N = %lld M = %lld\n", S, N, M );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }
}
