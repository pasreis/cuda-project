
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

#define BLOCK_SIZE 16

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

void makeIdentityMatrix(float* m, int n) {
	for (int row = 0; row < n; ++row) {
		for (int col = 0; col < n; ++col) {
			if (row == col) {
				m[row * n + col] = 1.0f;
			} else {
				m[row * n + col] = 0.0f;
			}
		}
	}
}


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

__global__
void doMulMatrix(float* a, float* b, float* c, int rowsA, int colsA, int colsB) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp = 0;

	if (col < colsB && row < rowsA) {
		for (int i = 0; i < colsA; ++i) {
			tmp += a[row * colsA + i] * b[i * colsB + col];
		}
		c[row * colsB + col] = tmp;
	}
}

__global__
void doDotProduct(float* a, float* b, float* c, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * blockDim.x;

	__shared__ float partialResult[BLOCK_SIZE * 2];

	unsigned int t = threadIdx.x;
	float tmp = 0.0f;

	for (int i = index; i < size; i += stride) {
		tmp += a[i] * b[i];
	}

	partialResult[t] = tmp;
	__syncthreads();
	unsigned int i = blockDim.x / 2;
	while (i != 0) {
		if (t < 1) {
			partialResult[t] += partialResult[t + i];
		}
		i /= 2;
	}
	if (threadIdx.x == 0) {
		c[0] = partialResult[0];
	}
}

__global__
void reduceNoDiagonal(float* m, float* I, int size, int i) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size) {
		if (row == i && row != col) {
			I[row * size + col] /= m[i * size + i];
			m[row * size + col] /= m[i * size + i];
		}
	}
}

__global__
void reduceDiagonal(float* m, float* I, int size, int i) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size) {
		if (row == col && row == i) {
			I[row * size + col] /= m[i * size + i];
			m[row * size + col] /= m[i * size + i];
		}
	}
}

__global__
void reduceLine(float* m, float* I, int size, int i) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size) {

		if (row != i) {
			I[row * size + col] -= I[i * size + col] * m[row * size + i];

			if (col != i) {
				m[row * size + col] -= m[i * size + col] * m[row * size + i];
			}
		}
	}
}

__global__
void makePivot(float* m, float* I, int size, int i) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < size && col < size) {
		if (row != i) {
			if (col == i) {
				m[row * size + col] = 0;
			}
		}
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

int mulMatrix(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC) {
	if (a == NULL || b == NULL || c == NULL || rowsA < 0 || colsA < 0 || rowsB < 0 || colsC < 0 || rowsC < 0) {
		printf("ERROR: mulMatrix, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: mulMatrix, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}



	float* d_a, *d_b, *d_c;
	size_t bytesA = (rowsA * colsA) * sizeof(float);
	size_t bytesB = (rowsB * colsC) * sizeof(float);
	size_t bytesC = (rowsC * colsB) * sizeof(float);

	size_t totalSize= bytesA + bytesB + bytesC;

	if (totalSize > totalMemory) {
		printf("ERROR: mulMatrix, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_a, bytesA);

	if (err != cudaSuccess) {
		printf("ERROR: matrixMul, error when trying to allocate memory for matrix A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytesA, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to move matrix A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytesB);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to allocate memory for matrix B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytesB, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to move matrix B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytesC);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to allocate memory for matrix C\n");
		return ERROR;
	}

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (colsA + BLOCK_SIZE - 1) / BLOCK_SIZE);

	doMulMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rowsA, colsA, colsB);

	cudaDeviceSynchronize();

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when multiplying the matrices\n");
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytesC, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when fetching matrix C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to free matrix A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to free matrix  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: mulMatrix, error when trying to free matrix C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

int addMatrix(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: addMatrix, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: addMatrix, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}



	float* d_a, *d_b, *d_c;
	size_t bytesA = (rows * cols) * sizeof(float);
	size_t bytesB = (rows * cols) * sizeof(float);
	size_t bytesC = (rows * cols) * sizeof(float);

	size_t totalSize= bytesA + bytesB + bytesC;

	if (totalSize > totalMemory) {
		printf("ERROR: addlMatrix, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_a, bytesA);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to allocate memory for matrix A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytesA, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to move matrix A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytesB);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to allocate memory for matrix B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytesB, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to move matrix B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytesC);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to allocate memory for matrix C\n");
		return ERROR;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = ((rows * cols) + threadsPerBlock - 1) / threadsPerBlock;

	doAddVector<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rows *  cols);

	err = cudaGetLastError();
	cudaDeviceSynchronize();



	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when adding the matrices %d\n", err);
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytesC, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when fetching matrix C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to free matrix A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to free matrix  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: addMatrix, error when trying to free matrix C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

int subMatrix(float* a, float* b, float* c, int rows, int cols) {
	if (a == NULL || b == NULL || c == NULL || rows < 0 || cols < 0) {
		printf("ERROR: subMatrix, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: subMatrix, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}



	float* d_a, *d_b, *d_c;
	size_t bytesA = (rows * cols) * sizeof(float);
	size_t bytesB = (rows * cols) * sizeof(float);
	size_t bytesC = (rows * cols) * sizeof(float);

	size_t totalSize= bytesA + bytesB + bytesC;

	if (totalSize > totalMemory) {
		printf("ERROR: subMatrix, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_a, bytesA);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to allocate memory for matrix A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytesA, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to move matrix A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytesB);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to allocate memory for matrix B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytesB, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to move matrix B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytesC);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to allocate memory for matrix C\n");
		return ERROR;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = ((rows * cols) + threadsPerBlock - 1) / threadsPerBlock;

	doSubVector<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, rows *  cols);

	err = cudaGetLastError();
	cudaDeviceSynchronize();



	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when subtracting the matrices %d\n", err);
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytesC, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when fetching matrix C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to free matrix A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to free matrix  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: subMatrix, error when trying to free matrix C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

int dotProduct(float* a, float* b, float* c, int size) {
	if (a == NULL || b == NULL || c == NULL || size < 0) {
		printf("ERROR: dotProduct, invalid values\n");
		return ERROR;
	}

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: dotProduct, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}



	float* d_a, *d_b, *d_c;
	size_t bytes = size * sizeof(float);

	size_t totalSize= bytes + bytes + bytes;

	if (totalSize > totalMemory) {
		printf("ERROR: dotProduct, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_a, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to allocate memory for vector A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to move vector A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_b, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to allocate memory for vector B\n");
		return ERROR;
	}

	err = cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to move vector B from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_c, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to allocate memory for vector C\n");
		return ERROR;
	}

	int threadsPerBlock = BLOCK_SIZE;
	int blocksPerGrid = BLOCK_SIZE;

	doDotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

	err = cudaGetLastError();
	cudaDeviceSynchronize();



	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when calculating the dot product %d\n", err);
		return ERROR;
	}

	err = cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when fetching vector C from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_a);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to free vector A from memory\n");
		return ERROR;
	}

	err = cudaFree(d_b);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to free vector  B from memory\n");
		return ERROR;
	}

	err = cudaFree(d_c);

	if (err != cudaSuccess) {
		printf("ERROR: dotProduct, error when trying to free vector C from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

void printMatrix(float* m, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf("%9.9f\t", m[i * cols + j]);
		}
		printf("\n");
	}
}
int inverseMatrix(float* m, float* inverse, int rows, int cols) {
	if (m == NULL || inverse == NULL || rows < 0 || cols < 0) {
		printf("ERROR: inverseMatrix, invalid values\n");
		return ERROR;
	}

	makeIdentityMatrix(inverse, rows);

	cudaError_t err;

	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to get the number of available devices\n");
		return ERROR;
	}

	cudaDeviceProp prop;
	int totalMemory = 0;

	for (int i = 0; i < deviceCount; ++i) {
		err = cudaGetDeviceProperties(&prop, i);

		if (err != cudaSuccess) {
			printf("ERROR: inverseMatrix, error when trying to get device properties\n");
			return ERROR;
		}

		totalMemory += prop.totalGlobalMem;
	}



	float* d_m, *d_inverse;
	int size = rows * cols;
	size_t bytes = size * sizeof(float);

	size_t totalSize= bytes + bytes + bytes;

	if (totalSize > totalMemory) {
		printf("ERROR: inverseMatrix, size is bigger than the total memory available in the system\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_m, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to allocate memory for matrix A\n");
		return ERROR;
	}

	err = cudaMemcpy(d_m, m, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to move matrix A from Host to Device\n");
		return ERROR;
	}

	err = cudaMalloc((float**) &d_inverse, bytes);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to allocate memory for inverse matrix\n");
		return ERROR;
	}

	err = cudaMemcpy(d_inverse, inverse, bytes, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when moving Identity Matrix to device\n");
		return ERROR;
	}

	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((size + BLOCK_SIZE -1) / BLOCK_SIZE, (size + BLOCK_SIZE -1) / BLOCK_SIZE);

	for (int i = 0; i < size; ++i) {
		reduceNoDiagonal<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_inverse, rows, i);
		cudaDeviceSynchronize();
		reduceDiagonal<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_inverse, rows, i);
		cudaDeviceSynchronize();
		reduceLine<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_inverse, rows, i);
		cudaDeviceSynchronize();
		makePivot<<<blocksPerGrid, threadsPerBlock>>>(d_m, d_inverse, rows, i);
		cudaDeviceSynchronize();
	}

	err = cudaGetLastError();




	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when calculating the inverse matrix%d\n", err);
		return ERROR;
	}

	err = cudaMemcpy(inverse, d_inverse, bytes, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when fetching the inverse matrix from Device to Host\n");
		return ERROR;
	}

	err = cudaFree(d_m);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to free matrix from memory\n");
		return ERROR;
	}

	err = cudaFree(d_inverse);

	if (err != cudaSuccess) {
		printf("ERROR: inverseMatrix, error when trying to free the inverse matrix from memory\n");
		return ERROR;
	}

	return SUCCESS;
}

void printVector(float* v, int size) {
	for (int i = 0; i < size; ++i) {
		printf("%9.9f\n", v[i]);
	}
}


