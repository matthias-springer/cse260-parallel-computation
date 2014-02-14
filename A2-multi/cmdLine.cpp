// Process command line arguments
// 
//
// Do not change the code in this file, as doing so
// could cause your submission to be graded incorrectly
//
#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include "types.h"
using namespace std;

void cmdLine(int argc, char *argv[], int& n, int &nreps, int &cores){
/// Command line arguments
 // Default value of the domain sizes
 static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"nreps", required_argument, 0, 'r'},
	{"cores", required_argument, 0, 'c'}
 };

 // Set default values
    n=16, nreps=5;

    // Process command line arguments
 int ac;
 for(ac=1;ac<argc;ac++) {
    int c;
    while ((c=getopt_long(argc,argv,"n:c:r:",long_options,NULL)) != -1){
        switch (c) {

	    // Size of the matrix
            case 'n':
                n = atoi(optarg);
                break;
	case 'c':
		cores = atoi(optarg);
		break;

	    // # of repititions
            case 'r':
                nreps = atoi(optarg);
		if (nreps < 1){
		    cerr << "nreps must be > 0. Exiting....\n\n";
		    exit(-1);
		}
                break;

	    // Error
            default:
                cout << "Usage: mmpy [-n <matrix dim>]  [-r <# repititions>]";
                cout << endl;
                exit(-1);
            }
    }
 }
}
