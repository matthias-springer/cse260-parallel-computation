#include "Bin.h"
#include "World.h"
#include <list>
#include <math.h>
#include <iostream>
#include <assert.h>

#ifdef _MPI_
// Conditional compilation for MPI
#include "mpi.h"
#endif

using namespace std;

extern double cutoff;
extern double cutoff2;
extern double mass;
extern double size;
extern double min_r;

// May come in handy when debugging
inline bool is_inside(double x, double y, const Rect& r)
{
	return ( x >= r.x_start && x <= r.x_end && y >= r.y_start && y <= r.y_end);
}

void Bin::AddParticle(particle_t* p)
{		
	binParticles.push_back(p);
}

void Bin::AddInboundParticle(particle_t* p)
{	
	printf("%f\n", p->x);
	inboundParticles.push_back(p);
}

// apply force from actors to reactors
void Bin::apply_forces(ParticleList& reactors, ParticleList& actors) {
	for( ParticleIterator reactor = reactors.begin(); reactor != reactors.end(); reactor++ ) {
	if ((*reactor)->tag < 0)
	    continue;

        for (ParticleIterator actor = actors.begin(); actor != actors.end(); actor++ ){
            if ( *reactor == *actor)
                continue;
            double dx = (*actor)->x - (*reactor)->x;
            double dy = (*actor)->y - (*reactor)->y;
            double r2 = dx * dx + dy * dy;
            if( r2 > cutoff * cutoff )
                continue;
            r2 = fmax( r2, min_r*min_r );
            double r = sqrt( r2 );

            //  very simple short-range repulsive force
            double coef = ( 1 - cutoff / r ) / r2 / mass;
            (*reactor)->ax += coef * dx;
            (*reactor)->ay += coef * dy;
        }
    }
}

// Apply forces to the bin
// The work divides into two parts as shown below
void Bin::apply_forces()
{
	// reset force
	for( ParticleIterator p = binParticles.begin(); p != binParticles.end(); p++ ) 
        (*p)->ax = (*p)->ay = 0.0;
	// 1. Apply forces from particles inside this bin
	//    to particles inside this Bin
	apply_forces(binParticles, binParticles);

	// 2. Apply forces from particles from neighboring bins
	//    to particles inside this Bin
	/*
	for(int dx=-1; dx <= 1; ++dx)
		for(int dy=-1; dy <= 1; ++dy) {
			if( dx == 0 && dy == 0)
				continue;
			int x = I + dx;
			int y = J + dy;
			if( !(x < 0 || y < 0 || x >= world->_nx || y >= world->_ny) )
				apply_forces(binParticles, world->bins[y*world->_nx + x].binParticles );
		}
	*/

	for (int i = 0; i < world->dimension; ++i) {
		if (world->my_rank > 0) {
			// have left neighbor => have left ghost cell column
			apply_forces(binParticles, world->left_ghost_bins[i].binParticles);
		}

		if (world->my_rank < world->cpu_count - 1) {
			// have right neighbor => have right ghost cell column
			apply_forces(binParticles, world->right_ghost_bins[i].binParticles);
		}
	}
}

//
//  integrate the ODE, advancing the positions of the particles
//
void Bin::move_particles(double dt)
{
	// update particles' locations and velocities
	for( ParticleIterator particle = binParticles.begin(); particle != binParticles.end(); particle++ ) {
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
	if ((*particle)->tag < 0)
	    continue;
        (*particle)->vx += (*particle)->ax * dt;
        (*particle)->vy += (*particle)->ay * dt;
        (*particle)->x  += (*particle)->vx * dt;
        (*particle)->y  += (*particle)->vy * dt;

    //
    //  bounce from walls
    //
        while( (*particle)->x < 0 || (*particle)->x > size ) {
            (*particle)->x  = (*particle)->x < 0 ? -(*particle)->x : 2*size-(*particle)->x;
            (*particle)->vx = -(*particle)->vx;
        }
        while( (*particle)->y < 0 || (*particle)->y > size ) {
            (*particle)->y  = (*particle)->y < 0 ? -(*particle)->y : 2*size-(*particle)->y;
            (*particle)->vy = -(*particle)->vy;
        }		
    }	
	
}

//
// We check each particle to see if it moved outside the present bin
// If, so we append to the inbound particle list for the new bin
//
void Bin::UpdateParticlesBin()
{
	int dummy = -1;
	MPI_Request bla;

	// Move to another Bin if needed
	for( ParticleIterator particle = binParticles.begin(); particle != binParticles.end(); ) {
		int i = (int) ((*particle)->x / world->binWidth);
		int j  = (int) ((*particle)->y / world->binHeight);
		//int newBin = j*world->_nx + i;
		
		int newI = i - world->left_bound;
		int newJ = j;
		int newBin = j*world->bins_length + i;

		// TODO: check if particle passed through multiple bins
		
		if (newI == -1) {
			// send to left neighbor
			// tag specifies the j index
			MPI_Isend((*particle), sizeof(particle_t), MPI_BYTE, world->my_rank - 1, j, MPI_COMM_WORLD, &bla);
			particle = binParticles.erase(particle);
		}
		else if (newI == world->bins_length) {
			// send to right neighbor
			MPI_Isend((*particle), sizeof(particle_t), MPI_BYTE,  world->my_rank + 1, j, MPI_COMM_WORLD, &bla);
			particle = binParticles.erase(particle);
		}
		else {
			if(i != I || j != J) {
				world->local_bins[newBin].AddInboundParticle((*particle));
				particle = binParticles.erase(particle);
			}
			else
				particle++;
			}
	}

	// send dummy particle to tell neighbors that all particles have been sent
	if (world->my_rank > 0) {
		// have left neigbor
		// negative tags are not allowed, so just use a j index that is too big
		MPI_Isend(&dummy, 1, MPI_INT, world->my_rank - 1, world->dimension, MPI_COMM_WORLD, &bla);
	}

	if (world->my_rank < world->cpu_count - 1) {
		// have right neighbor
		MPI_Isend(&dummy, 1, MPI_INT, world->my_rank + 1, world->dimension, MPI_COMM_WORLD, &bla);
	}

	// TODO: deallocate memory for sent particles
	
	// recevice
	if (world->my_rank > 0) {
		// receive from left neighbor
		MPI_Status status;

		do {
			particle_t * incoming_particle = (particle_t *) malloc(sizeof(particle_t));
			MPI_Recv(incoming_particle, sizeof(particle_t), MPI_BYTE, world->my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG < world->dimension) {
				// status.MPI_TAG is the j coordinate
				// receive from left neighbor => i=0
				int newBin = status.MPI_TAG*world->bins_length + 0;
				world->local_bins[newBin].AddInboundParticle(incoming_particle);	
			}
			else {
				free(incoming_particle);
			}

		} while (status.MPI_TAG != world->dimension);
	}

	if (world->my_rank < world->cpu_count - 1) {
		// has right neighbor
		MPI_Status status;

		do {
			particle_t * incoming_particle = (particle_t *) malloc(sizeof(particle_t));
			MPI_Recv(incoming_particle, sizeof(particle_t), MPI_BYTE, world->my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			if (status.MPI_TAG < world->dimension) {
				// receive from right neighbor => i=bin_length-1
				int newBin = status.MPI_TAG*world->bins_length + world->bins_length - 1;
				world->local_bins[newBin].AddInboundParticle(incoming_particle);
			}
			else {
				free(incoming_particle);
			}
		} while (status.MPI_TAG != world->dimension);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

//
// After we've updated all the inbound lists
// We then make a new pass, incorporating into this bin,
// any particles on the inbound list
//

void Bin::UpdateInboundParticles()
{
	for( ParticleIterator particle = inboundParticles.begin(); particle != inboundParticles.end(); particle++) {
		binParticles.push_back(*particle);
	}
	inboundParticles.clear();
}

void Bin::AddGhostParticle(particle_t * particle) {
	binParticles.push_back(particle);
}

void Bin::SendAsGhost(int targetRank, int tag) {
	MPI_Request bla;

	for (int i = 0; i < binParticles.size(); i++) {
		MPI_Isend(binParticles[i], sizeof(particle_t), MPI_BYTE, targetRank, tag, MPI_COMM_WORLD, &bla);
	}
}

