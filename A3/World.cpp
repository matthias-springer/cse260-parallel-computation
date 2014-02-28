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

World::World(double size, int nx, int ny, int np, int n, particle_t* particles, int num_cpus, int rank) : _size(size), _nx(nx), _ny(ny), _np(np), _n(n)
{
	my_rank = rank;
	cpu_count = num_cpus;

	binCount = nx * ny;
	bins = new Bin[binCount];
	binWidth = size / (double) nx;
	binHeight = size / (double) ny;
	for(int i=0; i < nx; ++i) {
            for(int j=0; j < ny; ++j) {			
                int binID = j*nx + i;	      
                bins[binID].I = i;
                bins[binID].J = j;
                bins[binID].world = this;
            }
	}
	SortParticles(n, particles);

		
	left_bound = (int) (ceil((1.0 * nx) / num_cpus) * my_rank);
	right_bound = (int) (min((int) (ceil((1.0 * nx) / num_cpus) * (my_rank + 1)), nx));

	bins_length = right_bound - left_bound;

  int left_bound_0 = (int) (ceil((1.0 * nx) / num_cpus) * 0);
  int right_bound_0 = (int) (min((int) (ceil((1.0 * nx) / num_cpus) * (0 + 1)), nx));

  max_bins_length = right_bound_0 - left_bound_0;

	dimension = nx;

	local_bins = new Bin[bins_length * nx];

	// copy local bins
	for (int i = 0; i < bins_length; ++i) {
		for (int j = 0; j < ny; ++j) {
			local_bins[j*bins_length + i] = bins[j*nx + i + bins_length*rank];
		}
	}

	// copy ghost bins
	if (left_bound > 0) {
		left_ghost_bins = new Bin[nx];

		for (int j = 0; j < ny; ++j) {
			left_ghost_bins[j] = bins[j*nx + bins_length*rank - 1];
		}
	}

	if (right_bound < nx) {
		right_ghost_bins = new Bin[nx];

		for (int j = 0; j < ny; ++j) {
			right_ghost_bins[j] = bins[j*nx + bins_length*rank + bins_length];
		}
	}
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
  for(int i=0; i < bins_length*dimension; ++i)
    local_bins[i].apply_forces();
 
}

//
// Move the particles in all the bins
//
void World::move_particles(double dt)
{
	MPI_Request bla;

  for(int i=0; i < bins_length*dimension; ++i)
    local_bins[i].move_particles(dt);

//printf("AFTER MOVE on %i\n", my_rank);
//
// After moving the particles, we check each particle
// to see if it moved outside its current bin
// If, so we append to the inbound partcle list for the new bin
// As written, this code is not threadsafe
//
  for(int i=0; i < bins_length*dimension; ++i)
    local_bins[i].UpdateParticlesBin();

//printf("AFTER UPDATE PARTICLES BIN on %i\n", my_rank);


//
// After we've updated all the inbound lists
// We then make a new pass, incorporating into this bin,
// any particles on the inbound list
// This work parallelizes
//

  for(int i=0; i < bins_length*dimension; ++i)
   local_bins[i].UpdateInboundParticles();

//printf("AFTER UPDATE INBOUND PARTICLES on %i\n", my_rank);

	// update ghost zones

	int dummy = -1;
	// send ghost_bins
	if (my_rank > 0) {
		// has left neighbor, send bins with i=0
		for (int j = 0; j < bins_length; ++j) {
			local_bins[j*bins_length + 0].SendAsGhost(my_rank - 1, j);
		}
		MPI_Isend(&dummy, 1, MPI_INT, my_rank - 1, dimension, MPI_COMM_WORLD, &bla);
	}

	if (my_rank < cpu_count - 1) {
		// has right neighbor, send bins with i=bin_length-1
		for (int j = 0; j < bins_length; ++j) {
			local_bins[j*bins_length + bins_length - 1].SendAsGhost(my_rank + 1, j);
		}
		MPI_Isend(&dummy, 1, MPI_INT, my_rank + 1, dimension, MPI_COMM_WORLD, &bla);
	}


	// receive ghost_bins
	// TODO: free memory
	left_ghost_bins = new Bin[dimension];
	right_ghost_bins = new Bin[dimension];

	if (my_rank > 0) {
		// receive from left neighbor, bins with i=bins_length-1
		MPI_Status status;
		
		do {
			particle_t * incoming_particle = (particle_t *) malloc(sizeof(particle_t));
			MPI_Recv(incoming_particle, sizeof(particle_t), MPI_BYTE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG < dimension) {
				left_ghost_bins[status.MPI_TAG].AddGhostParticle(incoming_particle);
			}
			else {
				free(incoming_particle);
			}
		} while (status.MPI_TAG != dimension);
	}
 
	if (my_rank < cpu_count - 1) {
		// receive from right neighbor, bins with i=0
		MPI_Status status;

		do {
			particle_t * incoming_particle = (particle_t *) malloc(sizeof(particle_t));
			MPI_Recv(incoming_particle, sizeof(particle_t), MPI_BYTE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG < dimension) {
				right_ghost_bins[status.MPI_TAG].AddGhostParticle(incoming_particle);
			}
			else {
				free(incoming_particle);
			}
		} while (status.MPI_TAG != dimension);
	}

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

void World::SimulateParticles(int nsteps, particle_t* particles, int n, int nt,  int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny, double dt ){
    for( int step = 0; step < nsteps; step++ ) {
		//	printf(" !!!!!!!!!!! BEGINNING OF LOOP !!!!!!!!!!!!!!\n");
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

	// TODO: send data and collect data from all bins
	int dummy = -1;
	
	MPI_Request bla;

	if (my_rank != 0) {
		for (int i = 0; i < bins_length*dimension; ++i) {
			// same functionality as sending as ghost
			local_bins[i].SendAsGhost(0, i);
		}
		MPI_Isend(&dummy, 1, MPI_INT, 0, dimension*dimension, MPI_COMM_WORLD, &bla);
	}
	else {
		bins = new Bin[dimension*dimension];

		// insert from local storage (rank 0)
		for (int i = 0; i < bins_length*dimension; ++i) {
			int particle_i = i % bins_length;
			int particle_j = i / bins_length;

			for (int j = 0; j < local_bins[i].binParticles.size(); ++j) {
				bins[particle_j*dimension + particle_i + left_bound].AddGhostParticle(local_bins[i].binParticles[j]);
			}
		}

		MPI_Status status;
		for (int r = 1; r < cpu_count; ++r) {
			int r_left_bound = (int) (ceil(dimension / cpu_count) * r);
			int r_right_bound = (int) (min((int) (ceil(dimension / cpu_count) * (r + 1)), dimension));
			int r_bins_length = r_right_bound - r_left_bound;

			do {
				particle_t * incoming_particle = (particle_t *) malloc(sizeof(particle_t));
				MPI_Recv(incoming_particle, sizeof(particle_t), MPI_BYTE, r, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				if (status.MPI_TAG == dimension*dimension) {
					// insert to global bins
					int particle_i = status.MPI_TAG % r_bins_length;
					int particle_j = status.MPI_TAG / r_bins_length;
					
					// just want to add a particle, use ghost method nvm
					bins[particle_j*dimension + particle_i + r_left_bound].AddGhostParticle(incoming_particle);
				}
				else {
					free(incoming_particle);
				}
			} while (status.MPI_TAG != dimension);
		}
	}
	
}


World::~World()
{
	delete [] bins;
}
