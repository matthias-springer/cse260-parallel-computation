#include "particle.h"
#include "Plotting.h"
extern double size;

#include <iostream>
using namespace std;

Plotter::Plotter() {
    gnu_pipe = popen("gnuplot -persist","w");
}

void Plotter::updatePlot(particle_t *particles, int n, int nIter, double umax, double vmax, double uL2, double vL2) {
    fprintf(gnu_pipe, "unset key\n");
    fprintf(gnu_pipe, "set xrange [0:%f]\n", size);
    fprintf(gnu_pipe, "set yrange [0:%f]\n", size);
    fprintf(gnu_pipe, "set title \"nIter = %d [(u,v) max=(%f,%f), (u,v) L2=(%f,%f)]\n",nIter, umax, vmax, uL2, vL2);
    fprintf(gnu_pipe, "set size square\n");
    fprintf(gnu_pipe, "plot \"-\" with points lt 1 pt 10 ps 1\n");

    // Write out the coordinates of the particles
    for( int i = 0; i < n; i++ ) {
        fprintf(gnu_pipe, "%.3f %.3f\n", particles[i].x, particles[i].y);
        }
        fprintf(gnu_pipe, "e\n");

    fflush(gnu_pipe);
#if 0
    if (simulation.plot_sleep) {
        sleep(simulation.plot_sleep);
    }
#endif

#if 0
      int dummy;
      scanf("%d",&dummy);
#endif
}

Plotter::~Plotter() {
    pclose(gnu_pipe);
}

