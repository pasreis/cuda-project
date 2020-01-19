
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

__global__
void doAddVector(float* a, float* b, float* c, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		c[i] = a[i] + b[i];
	}
}

__global__
void doSubVector(float* a, float* b, float* c, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		c[i] = a[i] - b[i];
	}
}

int addVector(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: addVector, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: vectorAdd, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}

	if (size > totalMemory) {
		printf("ERROR: vectorAdd, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	float* d_a, *d_b, *d_c;
	size_t bytes = size * sizeof(float);

	err = cudaMalloc((float**) &d_a, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to move vector A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to move vector B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector C\n");
		return ERROR;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	doAddVector<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when adding the vectors\n");
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when fetching vector C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

int subVector(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: addVector, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: vectorAdd, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}

	if (size > totalMemory) {
		printf("ERROR: vectorAdd, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	float* d_a, *d_b, *d_c;
	size_t bytes = size * sizeof(float);

	err = cudaMalloc((float**) &d_a, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to move vector A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to move vector B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to allocate memory for vector C\n");
		return ERROR;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	doSubVector<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when adding the vectors\n");
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when fetching vector C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: vectorAdd, error when trying to free vector C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

void printVector(float* v, int size) {
	for (int i = 0; i < size; ++i) {
		printf("%9.9f\n", v[i]);
	}
}

void printMatrix(float* m, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf("%9.9f\t", m[i * cols + j]);
		}
		printf("\n");
	}
}