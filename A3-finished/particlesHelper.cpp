// 
// Various helper functions
//
// Don't modify any code in this file
//
//
//
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "particle.h"
#include "common.h"
#include <iostream>
#include <time.h>
#ifdef _MPI_
// Conditional compilation for MPI
#include "mpi.h"
#endif


using namespace std;

double size;

double dt;

extern double density;
//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
    cout << "Simulation world size: " <<  size << endl;
}


//
//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p, int seed)
{
    long int sd;
    if (!seed)
        sd = time(NULL);
    else
        sd = seed;

    srand48( sd );
    cout << "Random seed is " << sd << endl;
        
    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;
    
    int *shuffle = new int[n];
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;
    
    for( int i = 0; i < n; i++ ) 
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];
        
        //
        //  distribute particles evenly to ensure proper spacing
        //
	double x = size*(1.+(k%sx))/(1+sx);
	double y = size*(1.+(k/sx))/(1+sy);
        p[i].x = x;
        p[i].y = y;

        //
        // Assign random velocities within a bound
        //
	p[i].vx = drand48()*2-1;
	p[i].vy = drand48()*2-1;
	p[i].tag = i;
    }
    delete [] shuffle;
}

// Output max velocity and L2 norm of x and y components of velocity
// L2 norm is sqrt(v_x^2 + v_y^2), where v_x and v_y are the
// x and y components of velocity, respectively
//
// This code only works for the single processor code
// For multiple cores, you will need to use VNorms and Norms defined below

void VelNorms(particle_t *particles, int n,
              double& uMax, double& vMax, double& uL2, double& vL2)
{
    uMax = vMax = -1e10;
    uL2 = vL2 = 0;
    for( int i = 0; i < n; i++ ) {
        double vx = fabs(particles[i].vx);
        double vy = fabs(particles[i].vy);
        uMax = fmax(uMax,vx);
        vMax = fmax(vMax,vy);
        uL2 += vx*vx;
        vL2 += vy*vy;
    }
    uL2 = sqrt (uL2/ (double) n);
    vL2 = sqrt (vL2/ (double) n);
}


// Use these two functions when computing the norm on multiple cores
// Compute max velocity and sum of the square of the
// x and y components of velocity (v_x^2 + v_y^2), where v_x and v_y are the
// x and y components of velocity, respectively
// The L2 norm is sqrt(v_x^2 + v_y^2)/n (global n) and we compute
// this with the help for Norms

void VNorms(particle_t *particles, int n,
              double& uMax, double& vMax, double& uL2, double& vL2)
{
    uMax = vMax = -1e10;
    uL2 = vL2 = 0;
    for( int i = 0; i < n; i++ ) {
        double vx = fabs(particles[i].vx);
        double vy = fabs(particles[i].vy);
        uMax = fmax(uMax,vx);
        vMax = fmax(vMax,vy);
        uL2 += vx*vx;
        vL2 += vy*vy;
    }
}

void Norms(int n, double& uL2, double& vL2)
{
    uL2 = sqrt (uL2/ (double) n);
    vL2 = sqrt (vL2/ (double) n);
}
//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

ostream & operator<<(ostream &os, const particle_t& p) {
    os << "Particle(x=" << p.x << ", y=" << p.y << ", vx=" << p.vx << ", vy=" << p.vy << ", ax=" << p.ax << ", ay=" << p.ay << ")";
    return os;
}

void list_particles(const particle_t* p, int n){
    for (int i=0; i<n; i++)
        cout << i << ": " << p[i] << endl;
    cout << endl;
}
void RepNorms(double uMax,double vMax,double uL2,double vL2){
    cout <<  "(u,v) Max = ";
    cout.precision(8);
    cout << scientific;
//    cout << fixed;
    cout << "( " << uMax << ", " << vMax << " )" << endl;
    cout <<  "(u,v) L2 = ";
    cout << "( " << uL2 << ", " << vL2 << " )" << endl;
}

void ReportPerformance(double Tp, int nt, int px, int py, int N, int nsteps, int nx,  double uMax, double vMax, double uL2, double vL2){
#ifdef _MPI_
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if (!myrank){
        std::cout << " Tp (sec)\t\tnt\tpx x py\tN\tnsteps nx       uMax, vMax, uL2, vL2\n";      
        std::cout << "@ " << Tp << "\t" <<   nt << "\t" << px << " x " << py << "\t" <<   N << "\t" <<  nsteps << "\t" << nx << "\t" <<   uMax << ", " <<   vMax << ", " <<  uL2 << ", " <<  vL2 << std::endl; 
    }
#else
        std::cout << " Tp (sec)\t\tnt\tN\tnsteps nx       uMax, vMax, uL2, vL2\n";      
        std::cout << "@ " << Tp << "\t" <<   nt << "\t" <<   N << "\t" <<  nsteps << "\t" << nx << "\t" <<   uMax << ", " <<   vMax << ", " <<  uL2 << ", " <<  vL2 << std::endl; 
#endif
}
