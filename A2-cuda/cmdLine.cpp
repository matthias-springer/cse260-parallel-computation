#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include "types.h"
using namespace std;

void cmdLine(int argc, char *argv[], int& n, int& reps, int& ntx, int& nty, _DOUBLE_ & eps, int& do_host, int& prefer_l1){

// Command line arguments, default settings

    n=8;
    reps = 10;

    // Threshold for comparison
    eps = 1.0e-6;

    // We don't do the computation on the host.
    do_host = 0;

    // We prefer Shared memory by default
    prefer_l1 = 0;


    // Ntx and Nty will be overriden by statically specified values
    // from the Make command line but are only useful when optimizing
    // for shared memory
#ifdef BLOCKDIM_X
    ntx = BLOCKDIM_X;
#else
    ntx = 8;
#endif

#ifdef BLOCKDIM_Y
    nty = BLOCKDIM_Y;
#else
    nty = 8;
#endif

 // Default value of the domain sizes
 static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"r", required_argument, 0, 'r'},
        {"ntx", required_argument, 0, 'x'},
        {"nty", required_argument, 0, 'y'},
        {"do_host", no_argument, 0, 'h'},
        {"eps", required_argument, 0, 'e'},
        {"l1", no_argument, 0, 'l'},
 };
    // Process command line arguments
 int ac;
 for(ac=1;ac<argc;ac++) {
    int c;
    while ((c=getopt_long(argc,argv,"n:r:x:y:he:l",long_options,NULL)) != -1){
        switch (c) {

	    // Size of the computational box
            case 'n':
                n = atoi(optarg);
                break;

            case 'r':
                reps = atoi(optarg);
                break;

	    // X thread block geometry
            case 'x':
#ifdef BLOCKDIM_X
                cout << " *** The thread block size is statically compiled.\n     Ignoring the X thread geometry command-line setting\n";
#else
                ntx = atoi(optarg);
#endif
                break;

	    // X thread block geometry
            case 'y':
#ifdef BLOCKDIM_Y
                cout << " *** The thread block size is statically compiled.\n      Ignoring the Y thread geometry command-line setting\n";
#else
                nty = atoi(optarg);
#endif
                break;

            case 'h':
                do_host = 1;
                break;

	    // comparison tolerance 
            case 'e':
#ifdef _DOUBLE
                sscanf(optarg,"%lf",&eps);
#else
                sscanf(optarg,"%f",&eps);
#endif
                break;


	    // Favor L1 cache (48 KB), else favor Shared memory
            case 'l':
                prefer_l1 = 1;
                break;

	    // Error
            default:
                printf("Usage: mm [-n <domain size>] [-r <reps>] [-x <x thread geometry> [-y <y thread geometry] [-e <epsilon>] [-h {do_host}] [-l  <prefer l1>]\n");
                exit(-1);
            }
    }
 }
}
