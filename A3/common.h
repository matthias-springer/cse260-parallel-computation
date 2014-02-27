#ifndef _COMMON_H__
#define _COMMON_H__

//
//  timing routines
//
double read_timer( );

//
//  argument processing routines
//
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );
void ABEND();
#endif
