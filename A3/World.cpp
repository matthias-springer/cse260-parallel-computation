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
#include <unistd.h>

extern particle_t * particles;

inline int World::cpu_x_of_particle(particle_t* particle) {
	return global_bin_x_of_particle(particle) / max_x_bins;
}

inline int World::cpu_y_of_particle(particle_t* particle) {
	return global_bin_y_of_particle(particle) / max_y_bins;
}

inline int World::local_bin_x_of_particle(particle_t* particle) {
	return global_bin_x_of_particle(particle) % max_x_bins;
}

inline int World::local_bin_y_of_particle(particle_t* particle) {
	return global_bin_y_of_particle(particle) % max_y_bins;
}

inline int World::cpu_of_particle(particle_t* particle) {
	return cpu_y_of_particle(particle) * thread_x_dim + cpu_x_of_particle(particle);
}

inline int World::local_bin_of_particle(particle_t* particle) {
	return local_bin_y_of_particle(particle) * bin_x_count + local_bin_x_of_particle(particle);
}

inline int World::global_bin_x_of_particle(particle_t* particle) {
	return (int) (particle->x / binWidth);
}

inline int World::global_bin_y_of_particle(particle_t* particle) {
	return (int) (particle->y / binHeight);
}

inline int World::bin_of_bin(int x, int y) {
	return y*_nx + x;
}

inline int World::cpu_of_cpu(int x, int y) {
	return y*thread_x_dim + x;
}

inline int World::cpu_of_bin(int x, int y) {
	int cpu_x = x / max_x_bins;
	int cpu_y = y / max_y_bins;
	return cpu_y * thread_x_dim + cpu_x;
}

inline void World::setup_thread() {
	my_rank_y = my_rank / thread_x_dim;
	my_rank_x = my_rank % thread_x_dim;

	max_x_bins = ceil(((float) _nx) / thread_x_dim);
	max_y_bins = ceil(((float) _ny) / thread_y_dim);

	bin_x_min = my_rank_x * max_x_bins;
	bin_x_max = min((my_rank_x + 1) * max_x_bins, _nx);
  bin_y_min = my_rank_y * max_y_bins;
  bin_y_max = min((my_rank_y + 1) * max_y_bins, _ny);

	bin_x_count = bin_x_max - bin_x_min;
	bin_y_count = bin_y_max - bin_y_min;
	bin_count = bin_y_count * bin_x_count;

	global_bin_count = _nx * _ny;

	thread_bin_x_min = new int[thread_x_dim];
	thread_bin_y_min = new int[thread_y_dim];
	thread_bin_x_max = new int[thread_x_dim];
	thread_bin_y_max = new int[thread_y_dim];

	for (int x = 0; x < thread_x_dim; ++x) {
		thread_bin_x_min[x] = x * max_x_bins;
		thread_bin_x_max[x] = min((x + 1) * max_x_bins, _nx);
	}

	for (int y = 0; y < thread_y_dim; ++y) {
	  thread_bin_y_min[y] = y * max_y_bins;
		thread_bin_y_max[y] = min((y + 1) * max_y_bins, _ny);
	}

	count_neighbors = 0;
	if (my_rank_x > 0) {
		count_neighbors++;

		if (my_rank_y > 0) count_neighbors++;
		if (my_rank_y < thread_y_dim - 1) count_neighbors++;
	}

	if (my_rank_y > 0) count_neighbors++;
	if (my_rank_y < thread_y_dim - 1) count_neighbors++;

	if (my_rank_x < thread_x_dim - 1) {
    count_neighbors++;

    if (my_rank_y > 0) count_neighbors++;
    if (my_rank_y < thread_y_dim - 1) count_neighbors++;
  }

	// setup ghost bin table
	ghost_bin_table = new fixed_int_8[global_bin_count];
	for (int x = 0; x < _nx; ++x) {
		for (int y = 0; y < _ny; ++y) {
			for (int i = 0; i < 3; i++) {
				ghost_bin_table[bin_of_bin(x, y)][i] = -1;
			}
			ghost_bin_table[bin_of_bin(x, y)][3] = 0;
		}
	}

	for (int cpu_x = 0; cpu_x < thread_x_dim; ++cpu_x) {
		for (int cpu_y = 0; cpu_y < thread_y_dim; ++cpu_y) {
			for (int x = thread_bin_x_min[cpu_x]; x < thread_bin_x_max[cpu_x]; ++x) {
				int bin_index = bin_of_bin(x, thread_bin_y_min[cpu_y]);
				if (cpu_y > 0) {
					// send to upper cpu
					ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x, cpu_y - 1);
					//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR A\n");
				}

				bin_index = bin_of_bin(x, thread_bin_y_max[cpu_y] - 1);
				if (cpu_y < thread_y_dim - 1) {
					// send to lower cpu
					ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x, cpu_y + 1);
					//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR B\n");
				}
			}

			for (int y = thread_bin_y_min[cpu_y]; y < thread_bin_y_max[cpu_y]; ++y) {
				int bin_index = bin_of_bin(thread_bin_x_min[cpu_x], y);
				if (cpu_x > 0) {
					// send to left cpu
					ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x - 1, cpu_y);
					//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR C\n");
				}

				bin_index = bin_of_bin(thread_bin_x_max[cpu_x] - 1, y);
				if (cpu_x < thread_x_dim - 1) {
					// send to right cpu
					ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x + 1, cpu_y);
					//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR D\n");
				}
			}

			// corner cases
			if (cpu_x > 0 && cpu_y > 0) {
				// send to left upper corner
				int bin_index = bin_of_bin(thread_bin_x_min[cpu_x], thread_bin_y_min[cpu_y]);
				ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x - 1, cpu_y - 1);
				//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR E\n");
			}

			if (cpu_x < thread_x_dim - 1 && cpu_y > 0) {
				// send to right upper corner
				int bin_index = bin_of_bin(thread_bin_x_max[cpu_x] - 1, thread_bin_y_min[cpu_y]);
				ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x + 1, cpu_y - 1);
				//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR F for cpu_x=%i, cpu_y=%i\n", cpu_x, cpu_y);
			}

			if (cpu_x < thread_x_dim - 1 && cpu_y < thread_y_dim - 1) {
				// send to lower right corner
				int bin_index = bin_of_bin(thread_bin_x_max[cpu_x] - 1, thread_bin_y_max[cpu_y] - 1);
				ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x + 1, cpu_y + 1);
				//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR G\n");
			}

			if (cpu_x > 0 && cpu_y < thread_y_dim - 1) {
				// send to lower left corner
				int bin_index = bin_of_bin(thread_bin_x_min[cpu_x], thread_bin_y_max[cpu_y] - 1);
				ghost_bin_table[bin_index][ghost_bin_table[bin_index][3]++] = cpu_of_cpu(cpu_x - 1, cpu_y + 1);
				//if (ghost_bin_table[bin_index][ghost_bin_table[bin_index][3] - 1] > 7) printf("ERR H\n");
			}
		}
	}
}

//#define BUFFER_SIZE 630

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
	setup_thread();

	for(int i=0; i < nx; ++i) {
            for(int j=0; j < ny; ++j) {			
                int binID = j*nx + i;	      
                bins[binID].I = i;
                bins[binID].J = j;
								bins[binID].my_rank = cpu_of_bin(i, j);
								bins[binID].my_rank_x = i / max_x_bins;
								bins[binID].my_rank_y = j / max_y_bins;
                bins[binID].world = this;
            }
	}
	SortParticles(n, particles);

	for (int i = 0; i < n; i++) {
		(particles + i)->tag = i;
	}

	num_buffers = (int) ceil(((float) n) / BUFFER_SIZE) + num_threads;
	num_buffers *= 4;	// TODO: think about this
	send_buffers = new send_buffer[num_buffers];
	send_buffer_index = new int[num_threads];

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

void World::flush_neighboring_send_buffers() {
	if (my_rank_x > 0) {
		flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y));

		if (my_rank_y > 0) {
			flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y - 1));
		}

		if (my_rank_y < thread_y_dim - 1) {
			flush_send_buffer(cpu_of_cpu(my_rank_x - 1, my_rank_y + 1));
		}
	}

	if (my_rank_y > 0) {
		flush_send_buffer(cpu_of_cpu(my_rank_x, my_rank_y - 1));
	}

	if (my_rank_y < thread_y_dim - 1) {
		flush_send_buffer(cpu_of_cpu(my_rank_x, my_rank_y + 1));
	}

  if (my_rank_x < thread_x_dim - 1) {
    flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y));

    if (my_rank_y > 0) {
      flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y - 1));
    }

    if (my_rank_y < thread_y_dim - 1) {
      flush_send_buffer(cpu_of_cpu(my_rank_x + 1, my_rank_y + 1));
    }
  }
}

void World::flush_send_buffers() {
	for (int i = 0; i < thread_count; i++) {
		if (i == my_rank) continue;
		flush_send_buffer(i);
	}
}

void World::reset_buffers() {
	for (int i = 0; i < num_buffers; ++i) {
		send_buffers[i].size = 0;
	}

	for (int i = 0; i < thread_count; ++i) {
		send_buffer_index[i] = i;
	}

	next_send_buffer = thread_count;
}

void World::receive_particles(int cpus) {
  int non_full_buffers = 0;
  send_buffer buffer;

  while (non_full_buffers < cpus - 1) {
    MPI_Status status;
    MPI_Recv(&buffer, sizeof(send_buffer), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    // put partices into memory and bins
    for (int i = 0; i < buffer.size; i++) {
      particle_t * particle =  buffer.particles + i;
      memcpy(particles + buffer.particles[i].tag, particle, sizeof(particle_t));

      int index_i = (int) (particle->x / binWidth);
      int index_j  = (int) (particle->y / binHeight);
      int newBin = index_j*_nx + index_i;

      bins[newBin].AddParticle(particles + buffer.particles[i].tag);
    }

    if (buffer.size < BUFFER_SIZE) {
      non_full_buffers++;
    }
  }
}

inline void World::receive_moving_particles() {
	receive_particles(thread_count);
}

inline void World::receive_moving_particles_from_neighbors() {
	receive_particles(count_neighbors + 1);
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


void World::send_particle(particle_t* particle, int target) {
		if (target == my_rank) {
			// send to ourselves
      int index_i = (int) (particle->x / binWidth);
      int index_j  = (int) (particle->y / binHeight);
      int newBin = index_j*_nx + index_i;

      bins[newBin].AddParticle(particle);
			return;
		}

    send_buffer* target_buffer = send_buffers + send_buffer_index[target];
    memcpy(target_buffer->particles + target_buffer->size++, particle, sizeof(particle_t));

    if (target_buffer->size == BUFFER_SIZE) {
      MPI_Request request;

      MPI_Isend(target_buffer, sizeof(send_buffer), MPI_BYTE, target, 0, MPI_COMM_WORLD, &request);
      send_buffer_index[target] = next_send_buffer++;
    }
}

inline void World::flush_send_buffer(int buffer) {
//		printf("[%i] flush send buffer: %i\n", my_rank, buffer);

    MPI_Request request;
    send_buffer* target_buffer = send_buffers + send_buffer_index[buffer];

    MPI_Isend(target_buffer, sizeof(int) + sizeof(particle_t) * target_buffer->size, MPI_BYTE, buffer, 0, MPI_COMM_WORLD, &request);
}

void World::check_send_ghost_particle(particle_t* particle, int target_rank_x, int target_rank_y, int bin_x, int bin_y) {
	for (int i = 0; i < 3; ++i) {
		int target_rank = ghost_bin_table[bin_of_bin(bin_x, bin_y)][i];
		
		if (target_rank != -1) {
			send_particle(particle, target_rank);
		}
	}

/*
	return; 
	int target_bin_x_min = thread_bin_x_min[target_rank_x];
	int target_bin_x_max = thread_bin_x_max[target_rank_x];
	int target_bin_y_min = thread_bin_y_min[target_rank_y];
	int target_bin_y_max = thread_bin_y_max[target_rank_y];

	// TODO: optimize: add if stmt with 4 checks in case we're not inside a ghost zone
	// TODO: optimize: array for cpu_of_cpu, ...
	// TODO: think about rearranging if statement nesting
	
	if (bin_x == target_bin_x_min && bin_y == target_bin_y_min) {
		// ul
		if (target_rank_x > 0) send_particle(particle, cpu_of_cpu(target_rank_x - 1, target_rank_y));
		if (target_rank_y > 0) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y - 1));
		if (target_rank_x > 0 && target_rank_y > 0) send_particle(particle, cpu_of_cpu(target_rank_x - 1, target_rank_y - 1));
	} else if (bin_x == target_bin_x_max-1 && bin_y == target_bin_y_min) {
		// ur
    if (target_rank_x < thread_x_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x + 1, target_rank_y));
    if (target_rank_y > 0) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y - 1));
    if (target_rank_x < thread_x_dim - 1 && target_rank_y > 0) send_particle(particle, cpu_of_cpu(target_rank_x + 1, target_rank_y - 1));
	} else if (bin_x == target_bin_x_max-1 && bin_y == target_bin_y_max-1) {
		// dr
    if (target_rank_x < thread_x_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x + 1, target_rank_y));
    if (target_rank_y < thread_y_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y + 1));
    if (target_rank_x < thread_x_dim - 1 && target_rank_y < thread_y_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x + 1, target_rank_y + 1));
	} else if (bin_x == target_bin_x_min && bin_y == target_bin_y_max-1) {
		// dl
    if (target_rank_x > 0) send_particle(particle, cpu_of_cpu(target_rank_x - 1, target_rank_y));
    if (target_rank_y < thread_y_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y + 1));
    if (target_rank_x > 0 && target_rank_y < thread_y_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x - 1, target_rank_y + 1));
	} else if (bin_y == target_bin_y_min) {
		// u
		if (target_rank_y > 0) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y - 1));
	} else if (bin_x == target_bin_x_max-1) {
		// r
		if (target_rank_x < thread_x_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x + 1, target_rank_y));
	} else if (bin_y == target_bin_y_max-1) {
		// d
		if (target_rank_y < thread_y_dim - 1) send_particle(particle, cpu_of_cpu(target_rank_x, target_rank_y + 1));
	} else if (bin_x == target_bin_x_min) {
		// l
		if (target_rank_x > 0) send_particle(particle, cpu_of_cpu(target_rank_x - 1, target_rank_y));
	}
*/
}

void World::send_ghost_particles() {
	// order: ul, u, up, l, r, dl, d, dr
	
	if (my_rank_x > 0 && my_rank_y > 0) {
		bins[bin_of_bin(bin_x_min, bin_y_min)].send_as_ghost(cpu_of_cpu(my_rank_x - 1, my_rank_y - 1));
	}

	if (my_rank_y > 0) {
		for (int x = bin_x_min; x < bin_x_max; ++x) {
			bins[bin_of_bin(x, bin_y_min)].send_as_ghost(cpu_of_cpu(my_rank_x, my_rank_y - 1));
		}
	}

	if (my_rank_y > 0 && my_rank_x < thread_x_dim - 1) {
		bins[bin_of_bin(bin_x_max - 1, bin_y_min)].send_as_ghost(cpu_of_cpu(my_rank_x + 1, my_rank_y - 1));
	}

	if (my_rank_x > 0) {
		for (int y = bin_y_min; y < bin_y_max; ++y) {
			bins[bin_of_bin(bin_x_min, y)].send_as_ghost(cpu_of_cpu(my_rank_x - 1, my_rank_y));
		}
	}

	if (my_rank_x < thread_x_dim - 1) { 
		for (int y = bin_y_min; y < bin_y_max; ++y) {
			bins[bin_of_bin(bin_x_max - 1, y)].send_as_ghost(cpu_of_cpu(my_rank_x + 1, my_rank_y));
		}
	}

	if (my_rank_x > 0 && my_rank_y < thread_y_dim - 1) {
		bins[bin_of_bin(bin_x_min, bin_y_max - 1)].send_as_ghost(cpu_of_cpu(my_rank_x - 1, my_rank_y + 1));
	}

	if (my_rank_y < thread_y_dim - 1) {
		for (int x = bin_x_min; x < bin_x_max; ++x) {
			bins[bin_of_bin(x, bin_y_max - 1)].send_as_ghost(cpu_of_cpu(my_rank_x, my_rank_y + 1));
		}
	}

	if (my_rank_x < thread_x_dim - 1 && my_rank_y < thread_y_dim - 1) {
		bins[bin_of_bin(bin_x_max - 1, bin_y_max - 1)].send_as_ghost(cpu_of_cpu(my_rank_x + 1, my_rank_y + 1));
	}
}

void World::clear_ghost_particles() {
	// TODO: optimize

	for (int x = -1; x < bin_x_count + 1; ++x) {
		int bin_index = bin_of_bin(x + bin_x_min, bin_y_min - 1);
		if (bin_index >= 0 && bin_index < global_bin_count) {
			bins[bin_index].binParticles.clear();
		}

		bin_index = bin_of_bin(x + bin_x_min, bin_y_max);
		if (bin_index >= 0 && bin_index < global_bin_count) {
			bins[bin_index].binParticles.clear();
		}

	}

	for (int y = -1; y < bin_y_count + 1; ++y) {
		int bin_index = bin_of_bin(bin_x_min - 1, y + bin_y_min);
		if (bin_index >= 0 && bin_index < global_bin_count) {
			bins[bin_index].binParticles.clear();
		}


		bin_index = bin_of_bin(bin_x_max, y + bin_y_min);
		if (bin_index >= 0 && bin_index < global_bin_count) {
			bins[bin_index].binParticles.clear();
		}
	}
}

void World::output_particle_stats() {
	int sum = 0;

	for (int x = bin_x_min; x < bin_x_max; ++x) {
		for (int y = bin_y_min; y < bin_y_max; ++y) {
			sum += bins[bin_of_bin(x,y)].binParticles.size();
		}
	}

	printf("STAT [%i]:     %i particles\n", my_rank, sum);
}

void World::SimulateParticles(int nsteps, particle_t* particles, int n, int nt,  int nplot, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave, int nx, int ny, double dt ){
    for( int step = 0; step < nsteps; step++ ) {
	//		printf("%i\n", step);

	//		output_particle_stats();

//		printf("[%i] Enter loop\n", my_rank);
    //
    //  compute forces
    //
	apply_forces();
    
    //
    //  move particles
    //
  clear_ghost_particles();
	move_particles(dt);
//	printf("After move_particles\n");

	send_ghost_particles();
	//flush_send_buffers();
	flush_neighboring_send_buffers();

	receive_moving_particles_from_neighbors();
	//receive_moving_particles();

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
//	 VelNorms(particles,n,uMax,vMax,uL2,vL2);
	
    //
    //  if we asked, save to a file every savefreq timesteps
    //
	if( fsave && (step%SAVEFREQ) == 0 ) {
	    save( fsave, n, particles );
	}
  }

	reset_buffers();


	if (my_rank == 0) {
		// receive
		for (int x = 0; x < _nx; ++x) {
			for (int y = 0; y < _ny; ++y) {
				if (x < bin_x_max && y < bin_y_max) continue;
				bins[bin_of_bin(x, y)].binParticles.clear();
			}
		}

		receive_moving_particles();
	}
	else {
		for (int x = bin_x_min; x < bin_x_max; ++x) {
			for (int y = bin_y_min; y <  bin_y_max; ++y) {
				int bin_index = bin_of_bin(x, y);
				bins[bin_index].send_as_ghost(0);
			}
		}

		flush_send_buffers();
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


World::~World()
{
	delete [] bins;
}
