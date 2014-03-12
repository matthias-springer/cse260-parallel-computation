#ifndef _WORLD_H
#define _WORLD_H

#include "particle.h"
#include "Plotting.h"

class Bin;

typedef int fixed_int_8[9];

class World
{
public:
	World(double size, int nx, int ny, int np, int n, particle_t* particles, int rank, int num_threads, int threads_x, int threads_y);
	~World();
	void SortParticles(int n, particle_t* particles);
	void apply_forces();
	void move_particles(double dt);
	void SimulateParticles(int nsteps, particle_t* particles, int n, int nt,  int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny, double dt );
	void receive_moving_particles();
	void reset_buffers();
	void send_ghost_particles();
	void clear_ghost_particles();

	int cpu_x_of_particle(particle_t* particle);
	int cpu_y_of_particle(particle_t* particle);
	int local_bin_x_of_particle(particle_t* particle);
	int local_bin_y_of_particle(particle_t* particle);
	int cpu_of_particle(particle_t* particle);
	int local_bin_of_particle(particle_t* particle);
	int global_bin_x_of_particle(particle_t* particle);
	int global_bin_y_of_particle(particle_t* particle);
	int cpu_of_bin(int x, int y);
	int bin_of_bin(int x, int y);
	int cpu_of_cpu(int x, int y);
	void send_particle(particle_t* particle, int target);
	void flush_send_buffers();
	void flush_send_buffer(int buffer);
	void flush_neighboring_send_buffers();
	void receive_particles(int cpus);
	void check_send_ghost_particle(particle_t* particle, int target_rank_x, int target_rank_y, int bin_x, int bin_y);
	void output_particle_stats();
	void receive_moving_particles_from_neighbors();
	void setup_thread();
	
	Bin* bins; // bins inside the world
	int binCount; // number of bins
	double _size; 
	int _nx, _ny, _np, _n;
	double binWidth, binHeight; //dimensions of each bin

	// thread-local variables
	int count_neighbors;
	int my_rank;
	int my_rank_x;
	int my_rank_y;
	int *thread_bin_x_min;
	int *thread_bin_y_min;
	int *thread_bin_x_max;
	int *thread_bin_y_max;

	int thread_count;
	int bin_x_min;
	int bin_x_max;
	int bin_y_min;
	int bin_y_max;
	int bin_x_count;
	int bin_y_count;
	int bin_count;
	int global_bin_count;

	int max_x_bins;
	int max_y_bins;

	int thread_x_dim;
	int thread_y_dim;

	fixed_int_8 *ghost_bin_table;
};

#endif
