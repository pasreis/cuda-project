/*
 * cuda-project-cpu.h
 *
 *  Created on: Jan 12, 2020
 *      Author: cuda-s10
 */

#ifndef CUDA_PROJECT_CPU_H_
#define CUDA_PROJECT_CPU_H_


void addVectorCPU(float* a, float* b, float* c, int size);
void subVectorCPU(float* a, float* b, float* c, int size);
void addMatrixCPU(float* a, float* b, float* c, int rows, int cols);
void subMatrixCPU(float* a, float* b, float* c, int rows, int cols);
void mulMatrixCPU(float* a, float* b, float* c, int rowsA, int colsA, int rowsB, int colsB, int rowsC, int colsC);
void dotProductCPU(float* a, float* b, float* c, int size);
void inverseMatrixCPU(float* a, float* inverse, int rows, int cols);



#endif /* CUDA_PROJECT_CPU_H_ */
