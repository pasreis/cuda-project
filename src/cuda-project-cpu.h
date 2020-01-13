/*
 * cuda-project-cpu.h
 *
 *  Created on: Jan 12, 2020
 *      Author: cuda-s10
 */

#ifndef CUDA_PROJECT_CPU_H_
#define CUDA_PROJECT_CPU_H_

#define SUCCESS 0
#define ERROR   1

int addVectorCPU(float* a, float* b, float* c, int size);
int subVectorCPU(float* a, float* b, float* c, int size);
int addMatrixCPU(float* a, float* b, float* c, int rows, int cols);
int subMatrixCPU(float* a, float* b, float* c, int rows, int cols);
int mulMatrixCPU(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC);
int dotProductCPU(float* a, float* b, float* c, int size);
int inverseMatrixCPU(float* a, float* inverse, int rows, int cols);



#endif /* CUDA_PROJECT_CPU_H_ */
