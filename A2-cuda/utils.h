#ifdef _UTILS_H
#else
#define _UTILS_H

void checkCUDAError(const char *msg);
void selectAndReport(int *major, int *minor);
#endif
