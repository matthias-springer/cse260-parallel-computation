//
// Parse command line arguments
//
//
// Don't modify any code in this file
//
//

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include "math.h"
#include "common.h"
#ifdef _MPI_
// Conditional compilation for MPI
#include "mpi.h"
#endif

using namespace std;

extern double dt;
int cmdLine( int argc, char **argv, int&n, int& nt, int& nsteps, int& nplot, int& sd, char** savename, int& nx, int& ny, int& px, int& py)
{
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        cout << "Options:\n";
        cout << "-h to see this help\n";
        cout << "-n <int> to set the number of particles [default = 8192]\n";
        cout << "-t <int> to set the number of timesteps i [1000]\n";
        cout << "-d to scale dt by sqrt(n)" << endl;
        cout << "-p <int> plot interval [off]\n";
        cout << "-s <int> to specify random seed [use time of day]\n";
        cout << "-o <filename> to specify the output file name\n";
        cout << "-nt <number of threads> \n";
        cout << "-px <processor geometry in the x direction>\n";
        cout << "-py <processor geometry in the y direction>\n";
	cout << "-M <number of rows and columns of bins> [default=1]> \n";
        return 0;
    }
    dt  =  0.0005;
    // Get command line arguments
    // Final argument is the default value
    n = read_int( argc, argv, "-n", 8192 );
    if( find_option( argc, argv, "-d" ) >= 0 ){
        dt /= sqrt((double) (n/1000));
        cout << "Scaling dt by sqrt(n)\n";
    }
    nsteps = read_int( argc, argv, "-t", 1000 );
    nplot = read_int( argc, argv, "-p", 0 );
    sd = read_int( argc, argv, "-s", 0 );
    *savename = read_string( argc, argv, "-o", NULL );
    nt = read_int( argc, argv, "-nt", 1 );
    px = read_int( argc, argv, "-px", 1 );
    py = read_int( argc, argv, "-py", 1 );
    nx = read_int( argc, argv, "-M", 20);
    ny = nx;

    int myrank=0;
#ifdef _MPI_
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if ((px * py) != nprocs){
        if (!myrank)
            cout << "\n *** The number of processes in the specified geometry (" << px*py << ")  is not the same as the number requested on the mpi launch line (" << nprocs << ")" << endl << endl;
        return(0);
    }

#else

    if ((px * py) > 1){
        if (!myrank)
            cout << "\n *** The number of processes in the specified geometry > 1, but you have not enabled MPI\n";
        return(0);
    }
#endif

    return(1);
}
