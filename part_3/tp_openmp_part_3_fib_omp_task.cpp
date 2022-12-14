/*
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include "omp.h"
#include <sys/time.h>

#ifndef FS
#define FS 38
#endif

using namespace std;

struct node {
   int data;
   int fibdata;
   struct node* next;
};

int fib(int n) {
   int x, y;
   if (n < 2) {
      return (n);
   } else {
      x = fib(n - 1);
      y = fib(n - 2);

	  return (x + y);
   }
}

void processwork(struct node* p) 
{
   int n;
   n = p->data;
   p->fibdata = fib(n);
}

struct node* init_list(struct node* p, int N) {
   int i;
   struct node* head = NULL;
   struct node* temp = NULL;
   
   head = (struct node*)malloc(sizeof(struct node));
   p = head;
   p->data = FS;
   p->fibdata = 0;
   for (i=0; i< N; i++) {
      temp  =  (struct node*)malloc(sizeof(struct node));
      p->next = temp;
      p = temp;
      p->data = FS + i + 1;
      p->fibdata = i+1;
   }
   p->next = NULL;
   return head;
}

int main(int argc, char *argv[]) {
   int N = 5;

   // Read command line arguments.
   for ( int i = 0; i < argc; i++ ) {
      if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_node" ) == 0 ) ) {
         N = atoi( argv[ ++i ] );
         printf( "  User num_node is %d\n", N );
      } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
         printf( "  Fib Options:\n" );
         printf( "  -num_node (-N) <int>:      Number of node computing fibonnaci numbers (by default 5)\n" );
         printf( "  -help (-h):            print this message\n\n" );
         exit( 1 );
      }
   }
    
   struct node *p=NULL;
   struct node *temp=NULL;
   struct node *head=NULL;

   printf("Process linked list\n");
   printf("  Each linked list node will be processed by function 'processwork()'\n");
   printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n",N,FS);      

   p = init_list(p, N);
   head = p;


   // Timer products.
   struct timeval begin, end;

   gettimeofday( &begin, NULL );

   # pragma omp parallel
   {
      # pragma omp single
      while (p != NULL) {
         # pragma omp task firstprivate(p)
         {
            processwork(p);
         }
         p = p->next;
      }
   }

   gettimeofday( &end, NULL );

   // Calculate time.
   double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
               1.0e-6 * ( end.tv_usec - begin.tv_usec );
               
   p = head;
   while (p != NULL) {
      printf("%d : %d\n",p->data, p->fibdata);
      temp = p->next;
      free (p);
      p = temp;
   }  
   free (p);

   printf("Compute Time: %f seconds\n", time);


   // output to file 
  ofstream myfile("stats.csv", ios::app);
  if (myfile.is_open())
  {
      myfile << "omp_tasks" << "," 
        << N << ","
        << std::setprecision(std::numeric_limits<double>::digits10) << time
        << endl;
      myfile.close();
  }
  else cerr<<"Unable to open file";

  return 0;

   return 0;
}

