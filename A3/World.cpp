// World manages the computational mesh, a list of particles
// sorted geographically into bins
// It also drives the simulator, calling other routines
// to do the computation
//
// A partition is a 2D array of bins, which stores the particles
//
#include <iostream>
#include "World.h"
#include "Bin.h"
#include "mpi.h"
#include <math.h>

extern particle_t * particles;

World::int cpu_x_of_particle(particle_t* particle) {
	return global_bin_x_of_particle(particle) / max_x_bins;
}

World::int cpu_y_of_particle(particle_t* particle) {
	return global_bin_y_of_particle(particle) / max_y_bins;
}

World::int local_bin_x_of_particle(particle_t* particle) {
	return global_bin_x_of_particle(particle) % max_x_bins;
}

World::int local_bin_y_of_particle(particle_t* particle) {
	return global_bin_y_of_particle(particle) % max_y_bins;
}

World::int cpu_of_particle(particle_t* particle) {
	return cpu_y_of_particle(particle) * thread_x_dim + cpu_x_of_particle(particle);
}

World::int local_bin_of_particle(particle_t* particle) {
	return local_bin_y_of_particle(particle) * bin_x_count + local_bin_x_of_particle(particle);
}

World::int global_bin_x_of_particle(particle_t* particle) {
	return (int) (particle->x / binWidth);
}

World::int global_bin_y_of_particle(particle_t* particle) {
	return (int) (particle->y / binHeight);
}

World::int bin_of_bin(int x, int y) {
	return y*_nx + x;
}

World::int cpu_of_cpu(int x, int y) {
	return y*thread_x_dim + x;
}

World::int cpu_of_bin(int x, int y) {
	int cpu_x = x / max_x_bins;
	int cpu_y = y / max_y_bins;
	return cpu_y * thread_x_dim + cpu_x;
}

World::void setup_thread() {
	my_rank_y = my_rank / thread_x_dim;
	my_rank_x = my_rank % thread_x_dim;

	max_x_bins = ceil(((float) _nx) / thread_x_dim);
	max_y_bins = ceil(((float) _ny) / thread_y_dim);

	bin_x_min = my_rank * max_x_bins;
	bin_x_max = min((my_rank + 1) * max_x_bins, _nx);
  bin_y_min = my_rank * max_y_bins;
  bin_y_max = min((my_rank + 1) * max_y_bins, _ny);

	bin_x_count = bin_x_max - bin_x_min;
	bin_y_count = bin_y_max - bin_y_min;
	bin_count = bin_y_count * bin_x_count;
}

#define BUFFER_SIZE 630

typedef struct send_buffer {
	int size;
	particle_t particles[BUFFER_SIZE];
} send_buffer;

send_buffer* send_buffers;
int* send_buffer_index;
int next_send_buffer;
int num_buffers;

World::World(double size, int nx, int ny, int np, int n, particle_t* particles, int rank, int num_threads, int threads_x, int threads_y) : _size(size), _nx(nx), _ny(ny), _np(np), _n(n), my_rank(rank), thread_count(num_threads), thread_x_dim(threads_x), thread_y_dim(threads_y)
{
	binCount = nx * ny;
	bins = new Bin[binCount];
	binWidth = size / (double) nx;
	binHeight = size / (double) ny;
	for(int i=0; i < nx; ++i) {
            for(int j=0; j < ny; ++j) {			
                int binID = j*nx + i;	      
                bins[binID].I = i;
                bins[binID].J = j;
								bins[binID].my_rank = cpu_of_bin(i, j);
                bins[binID].world = this;
            }
	}
	SortParticles(n, particles);

	for (int i = 0; i < n; i++) {
		(particles + i)->tag = i;
	}

	setup_thread();

	num_buffers = (int) ceil(((float) n) / BUFFER_SIZE) + num_threads;
	send_buffers = new send_buffer*[num_buffers];
	send_buffer_index = new int*[num_threads];

	for (int i = 0; i < num_threads; ++i) {
		send_buffer_index[i] = i;
	}

	next_send_buffer = num_threads;
}

//
// Sort only once at the start
//
void World::SortParticles(int n, particle_t* particles)
{
	for(int p=0; p < n; ++p) {
            int i = (int) (particles[p].x / binWidth);
            int j  = (int) (particles[p].y / binHeight);
            int binID = j* _nx + i;

            bins[binID].AddParticle(&particles[p]);
	}
}

//
// Apply forces to all the bins
//
void World::apply_forces()
{
	for (int x = bin_x_min; x < bin_x_max; x++) {
	  for(int y = bin_y_min; y < bin_y_max; y++) {
		  bins[y*_nx + x].apply_forces();
		}
	}
}

void World::send_buffers() {
	for (int i = 0; i < thread_count; i++) {
		if (i == my_rank) continue;

		MPI_Request request;
		send_buffer* target_buffer = send_buffers + send_buffer_index[i];

		MPI_Isend(target_buffer, sizeof(int) + sizeof(particle_t) * target_buffer->size, MPI_BYTE, i, 0, MPI_COMM_WORLD, &request);
	}
}

void World::reset_buffers() {
	for (int i = 0; i < num_buffers; ++i) {
		send_buffers[i]->size = 0;
	}

	for (int i = 0; i < num_threads; ++i) {
		send_buffer_index[i] = i;
	}

	next_send_buffer = thread_count;
}

void World::receive_moving_particles() {
	int non_full_buffers = 0;
	send_buffer buffer;

	do {
		MPI_Status status;
		MPI_Recv(&buffer, sizeof(send_buffer), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// put partices into memory and bins
		for (int i = 0; i < buffer.size; i++) {
			particle_t * particle =  buffer.particles + i;
			memcpy(particles + buffer.particles[i].tag, particle);

	    int i = (int) (particle->x / binWidth);
		  int j  = (int) (particle->y / binHeight);
			int newBin = j*world->_nx + i;

			bins[newBin].AddParticle(particle);
		}

		if (buffer.size < BUFFER_SIZE) {
			non_full_buffers++;
		}
	} while (non_full_buffers < thread_count - 1);

}

//
// Move the particles in all the bins
//
void World::move_particles(double dt)
{
  for (int x = bin_x_min; x < bin_x_max; x++) {
    for(int y = bin_y_min; y < bin_y_max; y++) {
	    bins[y*_nx + x].move_particles(dt);
		}
	}

//
// After moving the particles, we check each particle
// to see if it moved outside its current bin
// If, so we append to the inbound partcle list for the new bin
// As written, this code is not threadsafe
//
  for (int x = bin_x_min; x < bin_x_max; x++) {
    for(int y = bin_y_min; y < bin_y_max; y++) {
	    bins[y*_nx + x].UpdateParticlesBin();
		}
	}

//
// After we've updated all the inbound lists
// We then make a new pass, incorporating into this bin,
// any particles on the inbound list
// This work parallelizes
//
  for (int x = bin_x_min; x < bin_x_max; x++) {
    for(int y = bin_y_min; y < bin_y_max; y++) {
	    bins[y*_nx + x].UpdateInboundParticles();
		}
	}

	// TODO: sync ghost bins
	
#ifdef DEBUG
//
// May come in handy during debugging
//
  int particleCount = 0;
  for(int i=0; i < binCount; ++i) {
      particleCount += (int) bins[i].binParticles.size();
      for( unsigned int x=0; x <  bins[i].binParticles.size(); ++x) {
	particle_t* p = bins[i].binParticles[x];
	for( int j=0; j < binCount; ++j) {
	  if( i == j)
	    continue;
	  for(  unsigned int y=0; y < bins[j].binParticles.size(); ++y) {
	    if(p == bins[j].binParticles[y])
	      cout << "same particle detected in different bin\n" ;
	  }
	}
      }
    }
  if(particleCount != _n)
    cout << "particle count = " << particleCount << endl;
#endif
}

void World::send_ghost_particles() {
	// order: ul, u, up, l, r, dl, d, dr
	
	if (my_rank_x > 0 && my_rank_y > 0) {
		bins[bin_of_bin(bin_x_min, bin_y_min)].send_as_ghost(my_rank_x - 1, my_rank_y - 1);
		flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y - 1));
	}

	if (my_rank_y > 0) {
		for (int x = bin_x_min; x < bin_x_max; ++x) {
			bins[bin_of_bin(x, bin_y_min)].send_as_ghost(my_rank_x, my_rank_y - 1);
		}

		flush_send_buffer(cpu_of_cpu(my_rank_x, my_rank_y - 1));
	}

	if (my_rank_y > 0 && my_rank_x < thread_x_dim - 1) {
		bins[bin_of_bin(bin_x_min, bin_y_max - 1)].send_as_ghost(my_rank_x + 1, my_rank_y - 1);
		flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y - 1));
	}

	if (my_rank_x > 0) {
		for (int y = bin_y_min; y < bin_y_max; ++y) {
			bins[bin_of_bin(bin_x_min, y)].send_as_ghost(my_rank_x - 1, my_rank_y);
		}

		flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y));
	}

	if (my_rank_x < thread_x_dim - 1) { 
		for (int y = 0; y < bin_y_max; ++y) {
			bins[bin_of_bin(bin_x_max - 1, y)].send_as_ghost(my_rank_x + 1, my_rank_y);
		}

		flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y));
	}

	if (my_rank_x > 0 && my_rank_y < thread_y_dim - 1) {
		bins[bin_of_bin(bin_x_min, bin_y_max - 1)].send_as_ghost(my_rank_x - 1, my_rank_y + 1);
		flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y + 1));
	}

	if (my_rank_y < thread_y_dim - 1) {
		for (int x = bin_x_min; x < bin_x_max; ++x) {
			bins[bin_of_bin(x, bin_y_max - 1)].send_as_ghost(my_rank_x, my_rank_y + 1);
		}

		flush_send_buffer(cpu_of_cpu(my_rank_x, my_rank_y + 1));
	}

	if (my_rank_x < thread_x_dim - 1 && my_rank_y < thread_y_dim - 1) {
		bins[bin_of_bin(bin_x_max - 1, bin_y_max - 1)].send_as_ghost(my_rank_x + 1, my_rank_y + 1);
		flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y + 1));
	}
}

void World::SimulateParticles(int nsteps, particle_t* particles, int n, int nt,  int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny, double dt ){
    for( int step = 0; step < nsteps; step++ ) {
    //
    //  compute forces
    //
	apply_forces();

//     Debugging output
//      list_particles(particles,n);
    
    //
    //  move particles
    //
	move_particles(dt);

	send_buffers();
	receive_moving_particles();

	MPI_Barrier(MPI_COMM_WORLD);
	reset_buffers();

	if (nplot && ((step % nplot ) == 0)){

	// Computes the absolute maximum velocity 
	    VelNorms(particles,n,uMax,vMax,uL2,vL2);
	    plotter->updatePlot(particles,n,step,uMax,vMax,uL2,vL2);
	}

//
// Might come in handy when debugging
// prints out summary statistics every time step
//
	// VelNorms(particles,n,uMax,vMax,uL2,vL2);
	
    //
    //  if we asked, save to a file every savefreq timesteps
    //
	if( fsave && (step%SAVEFREQ) == 0 )
	    save( fsave, n, particles );
    }
}


World::~World()
{
	delete [] bins;
}
