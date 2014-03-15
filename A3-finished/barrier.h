#ifndef _BARRIER_H
#define _BARRIER_H

/****************************************************************
*								*
* Author:  Pietro Cicotti 		                        *
* Ported to C++ NT by Scott B. Baden, 10/25/13                  *    
* This is a basic linear time barrier implementation            *
* This code is a starting point for more efficient barriers     *
*								*
*****************************************************************/

#include <mutex>
#include <cassert>


using namespace std;

#ifdef LOG_BARRIER
class barrier
{
public:
    // Fill in your definition of the barrier constructor
    barrier(int NT = 2) { }
    // Fill in your definition of the barrier synchronization function
    void bsync(int TID) {
    }
};
#else
class barrier
{
    int count;
    int _NT;
    mutex arrival;
    mutex departure;

public:

    barrier(int NT = 2) {
        arrival.unlock();
        departure.lock();
        count = 0;
        _NT = NT;
    }

    void bsync() {
        arrival.lock();
        ++count; // atomically count the waiting NT

        if(count < _NT)
            arrival.unlock();
        else // last  processor enables all to go	
            departure.unlock();

        departure.lock();
        --count;  // atomically decrement
        if(count > 0)
            departure.unlock();
        else
            arrival.unlock();
    }

};
#endif
#endif

