#include "stdio.h"

#define CUDA_ERROR_CHECK(error) if (error != cudaSuccess) { fprintf(stderr, "%s %i %s: %s.\n", __FILE__, __LINE__, cudaGetErrorName(error), cudaGetErrorString(error)); exit(error); }
