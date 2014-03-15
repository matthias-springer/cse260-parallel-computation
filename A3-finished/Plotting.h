#ifndef __PLOTTING_H
#define __PLOTTING_H
#include <stdlib.h>
#include <cstdio>

#include "particle.h"

class Plotter {
public:
    Plotter();
    ~Plotter();
    void updatePlot(particle_t *particles, int n, int step, double umax, double vmax, double uL2, double vL2);


private:
    FILE *gnu_pipe;
};

#endif
