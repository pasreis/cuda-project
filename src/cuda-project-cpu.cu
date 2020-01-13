
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "cuda-project-cpu.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

__host__
int addVectorCPU(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: addVectorCPU invalid values\n");
		return ERROR;
	}

	for (int i = 0; i < size; ++i) {
		c[i] = a[i] + b[i];
	}

	return SUCCESS;
}

