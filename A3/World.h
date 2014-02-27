#ifndef _WORLD_H
#define _WORLD_H

#include "particle.h"
#include "Plotting.h"

class Bin;

class World
{
public:
	World(double size, int nx, int ny, int np, int n, particle_t* particles, int num_cpus, int rank);
	~World();
	void SortParticles(int n, particle_t* particles);
	void apply_forces();
	void move_particles(double dt);
	void SimulateParticles(int nsteps, particle_t* particles, int n, int nt,  int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny, double dt );

	Bin* bins; // bins inside the world
	Bin* local_bins;

	Bin* left_ghost_bins;
	Bin* right_ghost_bins;

	int left_bound;
	int right_bound;
	int bins_length;

	int dimension;
	int my_rank;
	int cpu_count;

	int binCount; // number of bins
	double _size; 
	int _nx, _ny, _np, _n;
	double binWidth, binHeight; //dimensions of each bin
};

#endif
