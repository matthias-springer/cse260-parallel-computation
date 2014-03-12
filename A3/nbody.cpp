#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include "particle.h"
#include "common.h"
#include "Plotting.h"
#include "World.h"
       #include <sys/types.h>
       #include <unistd.h>

#ifdef _MPI_
// Conditional compilation for MPI
#include "mpi.h"
#endif

using namespace std;


// This is how to conditionally compile for OpenMP
#ifdef  _OPENMP
#include <omp.h>
#endif
void ABEND();

using namespace std;

extern double dt;
extern double size;

//
//  Tuned constants
//
double density = 0.0005;
double  mass =    0.01;
double cutoff =  0.01;
double min_r  =  (cutoff/100);
particle_t *particles;

int cmdLine( int argc, char **argv, int&n, int& nt, int& nsteps, int& ntlot, int& sd, char** savename, int& nx, int& ny, int& px, int& py);
void SimulateParticles(int nsteps, particle_t *particles, int n, int nt, int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny );
void RepNorms(double uMax,double vMax,double uL2,double vL2);
void ReportPerformance(double Tp, int nt, int px, int py, int N, int nsteps, int nx, double uMax, double vMax, double uL2, double vL2);

int main( int argc, char **argv )
{  
		int nt;     // number of threads
    int n;      // # of particles
    int nsteps; // # of timesteps
    int nplot;  // Plotting frequency
    int sd;     // Random seed
    char *savename;     // Name of file to save output
    int nx, ny; // number of columns and rows of bins
                // nx = ny in this simulation, though local bounds
                // may differ
    int px, py; // processor geometry

///printf("PID %d ready for attach\n", getpid());

#ifdef _MPI_
 MPI_Init(&argc,&argv);
#endif


#ifdef _MPI_
    int local_OK =  cmdLine( argc, argv, n, nt, nsteps, nplot, sd,  &savename, nx, ny, px, py);
    int OK;
    MPI_Reduce(&local_OK, &OK, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);
#else
    int OK =  cmdLine( argc, argv, n, nt, nsteps, nplot, sd,  &savename, nx, ny, px, py);
#endif

    // If there was a parsing error, exit
//    if (!OK)
//        ABEND();

    set_size( n );

    //sanity check
    if( size / (double) nx < cutoff ) {
        cout << "nx is too large" << endl;
        ABEND();
    }

    int nprocs=1, myrank=0;
#ifdef _MPI_
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif
		//sleep(30);

    FILE *fsave;
    // Only process 0 saves the particle positions.
    if ( !myrank) {
        cout << endl << "dt: " << dt << endl;
    
        fsave = savename ? fopen( savename, "w" ) : NULL;
    }

    particles = new particle_t[ n ];
    assert(particles);
    
    init_particles( n, particles,sd);
    double uMax, vMax, uL2, vL2;
    Plotter *plotter = NULL;
    if (nplot){
        plotter = new Plotter();
        assert(plotter);
        VelNorms(particles,n,uMax, vMax, uL2, vL2);
        plotter->updatePlot(particles,n,0,uMax,vMax,uL2,vL2);
    }
    if ( !myrank) {
        cout << "# particles : " << n << endl;
        cout << "Nsteps: " << nsteps << endl;
        cout << "Partition: " << nx << " x "<< ny << endl;
        cout << "Nt= " << nt << endl;
    }
#ifdef _MPI_
    if ( !myrank) {
        cout << "Processor geometry: " << px << " x " << py << endl;
    }
#endif
    //
    // Bin the particles into nx by ny regions
    //
		//printf("Creating world on rank %i\n", myrank);
    World world(size, nx, ny, nt, n, particles, myrank, nprocs, px, py);
  
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    world.SimulateParticles(nsteps,particles,n,nt, nplot, uMax, vMax, uL2, vL2, plotter, fsave, nx, ny, dt );
    simulation_time = read_timer( ) - simulation_time;
   
    if ( !myrank) {
        cout << endl;
        cout <<  "n = " << n << ", nsteps = " << nsteps << endl;
    }
    VelNorms(particles,n,uMax,vMax,uL2,vL2);
		
		if (!myrank)
	    RepNorms(uMax,vMax,uL2,vL2);
    if ( !myrank) {
        cout <<  "Running time = " << simulation_time << " sec.\n";
    }
    ReportPerformance( simulation_time, nt, px, py, n, nsteps, nx, uMax,  vMax,  uL2,  vL2);

    if ( !myrank) {
        if( fsave )
            fclose( fsave );
    }
    
// Only process 0 handles plotting
    if ( !myrank) {
        if (nplot)
            delete plotter;
    }
    delete [ ] particles;

#ifdef _MPI_
    MPI_Finalize();
#endif
    
    return 0;
}
