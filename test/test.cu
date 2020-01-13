
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

#include "../src/cuda-project.h"
#include "../src/cuda-project-cpu.h"

#define PASSED 1
#define FAILED 0

#define CPU_TEST_VECTOR_SIZE 3
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

int testAddVectorCPU() {
	printf("Testing addVectorCPU with values:\n");

	float a[CPU_TEST_VECTOR_SIZE] = { 1.0f, 2.0f, 3.0f };
	float b[CPU_TEST_VECTOR_SIZE] = { 4.0f, 5.0f, 6.0f };
	float c[CPU_TEST_VECTOR_SIZE];
	float c_expected[CPU_TEST_VECTOR_SIZE] = { 5.0f, 7.0f, 9.0f };

	printf("A:\n");
	printVector(a, CPU_TEST_VECTOR_SIZE);
	printf("\nB:\n");
	printVector(b, CPU_TEST_VECTOR_SIZE);
	printf("\nSize: %d\n", CPU_TEST_VECTOR_SIZE);

	if (addVectorCPU(a, b, c, CPU_TEST_VECTOR_SIZE) != SUCCESS) return FAILED;

	for (int i = 0; i < CPU_TEST_VECTOR_SIZE; ++i) {
		if (c[i] != c_expected[i]) {
			printf("ERROR: addVector CPU, expected %f but got %f instead!\n", c_expected[i], c[i]);
			return FAILED;
		}
	}

	printf("Testing addVectorCPU with NULL vector A\n");
	if (addVectorCPU(NULL, b, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with NULL vector B\n");
	if (addVectorCPU(a, NULL, c, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with NULL vector C\n");
	if (addVectorCPU(a, b, NULL, CPU_TEST_VECTOR_SIZE) != ERROR) return FAILED;

	printf("Testing addVectorCPU with negative size\n");
	if (addVectorCPU(a, b, c, -1) != ERROR) return FAILED;

	return PASSED;
}

/**
 * This main function is responsible for running the necessary tests
 * to ensure the correctness of the solution
 */
int main(int argc, char **argv)
{
	if (testAddVectorCPU() == PASSED) printf("TEST PASSED!\n"); else printf("TEST FAILED!\n");
	return 0;
}
