#ifndef _BIN_H
#define _BIN_H

#include "particle.h"
#include <vector>
#include <list>

using namespace std;

// We use ParticleLists to represent the list of particles
// owned by a bin
//
// We use a particle_t* to store the global list of particles
//
// Both refer to the same set of data in memory
//
typedef vector<particle_t*> ParticleList;
typedef ParticleList::iterator ParticleIterator;

struct Rect
{
	double x_start, x_end, y_start, y_end;
};

class World;

class Bin
{
public:  
  ParticleList binParticles; // the particles inside this bin
  ParticleList inboundParticles; // the particles coming into this bin
  				 // We use this list to keep track of
				 // all incoming migrating particles
  int I, J;  	// Index for this bin
	int my_rank;
	int my_rank_x;
	int my_rank_y;

  World* world; // the world object used to access other bins
    
  void AddParticle(particle_t* p);
  void AddInboundParticle(particle_t* p);
  void apply_forces();
  void apply_forces(ParticleList& reactors, ParticleList& actors);
  void move_particles(double dt);
  void UpdateParticlesBin();
  void UpdateInboundParticles();
	void send_as_ghost(int target);
	void binParticlesP_clear();
};


#endif
