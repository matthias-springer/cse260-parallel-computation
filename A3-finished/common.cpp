#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
using namespace std;
#include <sys/time.h>

#ifdef _MPI_
// Conditional compilation for MPI
#include "mpi.h"
#endif


//
//  timer
//
double read_timer( ) {

#ifdef _MPI_
    return MPI_Wtime();
#else
    static bool initialized = false;
    static timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
#endif
}


//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}

void ABEND()
{
   cout.flush();
   cerr.flush();
#ifdef _MPI_
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
#endif
   exit(-1);
}
