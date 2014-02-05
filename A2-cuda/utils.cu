// Some useful utilities
// system includes
#include <stdio.h>
#include <assert.h>
#include <cuda.h>


// External function definitions

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
    // When there is more than one device,
    // select the fastest device and report characteristics

void selectAndReport(int *major, int *minor)
{
        int best_gpu = 0;
        // Use the device with the most cores
        int number_of_devices;
	cudaGetDeviceCount(&number_of_devices);
	checkCUDAError("Get Count");
        printf("\n%d Devices\n",number_of_devices);
        if (number_of_devices > 1) {
            int max_cores = 0;
            int device_number;
            for (device_number = 0; device_number < number_of_devices;
        device_number++) {
                cudaDeviceProp device_properties;
                cudaGetDeviceProperties(&device_properties, device_number);
                printf("Device # %d has %d cores\n",device_number, device_properties.multiProcessorCount);
                double gb = 1024*1024*1024;
                printf("Device # %d has %f GB global memory\n",device_number, ((double)device_properties.totalGlobalMem)/gb);
                if (max_cores < device_properties.multiProcessorCount) {
                    max_cores = device_properties.multiProcessorCount;
                    best_gpu = device_number;
                }
            }

            printf("\n *** Best GPU is: %d\n",best_gpu);
            cudaSetDevice(best_gpu);
            checkCUDAError("Can't set device\n");

        }
        printf("\n");
// get number of SMs on this GPU
        int devID;
        cudaGetDevice(&devID);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devID);
        printf("Device %d has %d cores\n", best_gpu, deviceProp.multiProcessorCount);
        double gb = 1024*1024*1024;
        printf("Device # %d has %f GB global memory\n",best_gpu, ((double)deviceProp.totalGlobalMem)/gb);

        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            printf("There is no device supporting CUDA.\n");
            cudaThreadExit();
        } else {
            printf("Device is a %s, capability: %d.%d\n",  deviceProp.name, deviceProp.major, deviceProp.minor);
	    *major  = deviceProp.major;
	    *minor = deviceProp.minor;
        } 

        int driverVersion, runtimeVersion;
	assert(cudaSuccess == cudaDriverGetVersion(&driverVersion));
	assert(cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion));
        printf("CUDA Driver version: %d, runtime version: %d\n\n", driverVersion, runtimeVersion);

}
