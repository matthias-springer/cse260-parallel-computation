#ifndef __PARTICLE_H
#define __PARTICLE_H
#include <ostream>
using std::ostream;


//
// particle data structure
//
typedef struct 
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
  int tag;
} particle_t;



//
//  saving parameters
//
const int NSTEPS = 1000;
const int SAVEFREQ = 10;

//
//  simulation routines
//
void set_size( int n );
void init_particles( int n, particle_t *p , int sd);
// void apply_forces( particle_t* particles, int n);
// void move_particles( particle_t* particles, int n); 

//
//  I/O routines
//
FILE *open_save( char *filename, int n );
void save( FILE *f, int n, particle_t *p );


ostream & operator<<(ostream &os, const particle_t& p);
void list_particles(const particle_t* p, int n);

//
// Reporting routines
//
void VelNorms(particle_t *particles, int n,
              double& uMax, double& vMax, double& uL2, double& vL2);

void VNorms(particle_t *particles, int n,
              double& uMax, double& vMax, double& uL2, double& vL2);

void Norms(int n, double& uL2, double& vL2);

#endif
