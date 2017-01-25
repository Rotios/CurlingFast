//(c) 2017 John Freeman and Jose Rivas

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 32
#define X(r, c) ((r+1) * (r * 0.5)) + c  

void printTable(char *table, int length) {
    char *CPUTable;

    CPUTable = (char *)malloc(length * sizeof(char));
    cudaMemcpy(CPUTable, table, length * sizeof(char), cudaMemcpyDeviceToHost);

    for (int i(0); i < length; ++i) {
        printf("%c ", CPUTable[i]);
    }

    printf("\n");
    free(CPUTable);
}

void printBoolArray(bool *comparisons, int size) {

    bool *comp;
    comp = (bool *)malloc(size * sizeof(bool));

    cudaMemcpy(comp, comparisons, size * sizeof(bool), cudaMemcpyDeviceToHost);
    printf("comparisons =  ");
    for (int i(0); i < size; ++i) {
        printf("%d ", comp[i]);
    }

    printf("\n");
}

__global__ void maxCompare(char *a, bool *check, int *size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx == idy) { return; }

    int xval = a[idx];
    int yval = a[idy];

    if (xval < yval) {
        check[idx] = false;
    }
}

__global__ void cudaMax(char *temps, bool *comparisons, int* size, char *sequence) {
    int idx = blockIdx.x;

    if (comparisons[idx]) {
        sequence[(*size)++] = temps[idx];
    }
    comparisons[idx] = true;
}

// Magic if it works
__global__ void fillRow(char *table, char *sequence, int *size) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = *size - 1;
    int pastSequencePos = row - (column + 1);
    int position = X(row, column);
    int position2 = (pastSequencePos > column) * X(pastSequencePos, column);

    table[position] += '1' + (sequence[pastSequencePos] == sequence[row]) * (((pastSequencePos > column) * (table[position2] - '0')) + (pastSequencePos <= column));
}

__global__ void fillInitialRow(char *table, char *sequence, int *index) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = *index;
    int pastSequencePos = row - (column + 1);
    int position = X(row, column);
    int position2 = (pastSequencePos > column) * X(pastSequencePos, column);

    table[position] += '1' + (sequence[pastSequencePos] == sequence[row]) * (((pastSequencePos > column) * (table[position2] - '0')) + (pastSequencePos <= column));
}

void initializeTable(char *table, char *sequence, int seqLength) {

    int *index;
    cudaMalloc((void **)&index, sizeof(int));

    for (int i(0); i < seqLength; ++i) {
        cudaMemcpy(index, (void *)&i, sizeof(int), cudaMemcpyHostToDevice);
        fillInitialRow << < dim3(i + 1, 1), 1 >> > (table, sequence, index);
    }

    cudaFree(index);
}

__global__ void cudaMin(char *table, char *temps, bool *comparisons, int *size) {
    int colIdx = blockIdx.y;
    int rowIdx = blockIdx.x;

    int seqLength = *size;

    int boolId = colIdx * seqLength + rowIdx;
    int tabRowIdx = seqLength - 1 - rowIdx;

    int idx = X(tabRowIdx, colIdx);

    if (comparisons[boolId] && rowIdx <= colIdx) {
        temps[colIdx] = table[idx];
    }

    comparisons[boolId] = true;
}

__global__ void minCompare(char *table, char *temps, bool *comparisons, int *size) {

    int colIdx = blockIdx.x;

    int rowIdx = threadIdx.x;
    int rowIdy = threadIdx.y;

    int seqLength = *size;

    // Work backwards on the table rows
    int tabRowIdx = seqLength - 1 - rowIdx;
    int tabRowIdy = seqLength - 1 - rowIdy;

    //// Get the index of the value we are looking at
    bool test = (rowIdx <= colIdx);
    int tabIdx = test * X(tabRowIdx, colIdx);
    int tabIdy = X(tabRowIdy, colIdx);

    int boolId = colIdx * seqLength + rowIdx;
    int xval = (table[tabIdx] - '0');
    int yval = (test && rowIdy <= colIdx) *  (table[tabIdy] - '0');

    if (yval == 0 || yval == xval) {

    }
    else if (xval == 0 || xval > yval) {
        comparisons[boolId] = false;
    }

}

void findCurl(char *table, char *sequence, char *temps, bool *comparisons, int *size, int *curl, int& seqLength) {
    int numBlocks = seqLength >> 1;
    minCompare << < numBlocks, dim3(numBlocks, numBlocks) >> > (table, temps, comparisons, size);
    cudaMin << < dim3(numBlocks, numBlocks), 1 >> > (table, temps, comparisons, size);
    maxCompare << < dim3(numBlocks, numBlocks), 1 >> > (temps, comparisons, size);
    cudaMax << < dim3(numBlocks, numBlocks), 1 >> > (temps, comparisons, size, sequence);
}


int main() {
    char *table;
    char *sequence;
    char *temps;
    bool *comparisons;
    int *size;
    int *cuda_capacity;
    int *curl;
    int capacity = INITIAL_CAPACITY;

    // ((capacity + 1) * capacity) / 2 for table
    int table_size = (((capacity + 1) * capacity) / 2);

    // size needed for bool arrays in min and max functions
    int compare_size = (capacity / 2) * (capacity / 2);
    char *seq = (char *)malloc(capacity * sizeof(char));

    cudaMalloc((void**)&table, table_size * sizeof(char));
    cudaMalloc((void**)&sequence, capacity * sizeof(char));
    cudaMalloc((void**)&temps, capacity * sizeof(char));
    cudaMalloc((void**)&comparisons, compare_size * sizeof(bool));
    cudaMalloc((void**)&size, sizeof(int));
    cudaMalloc((void**)&cuda_capacity, sizeof(int));
    cudaMalloc((void**)&curl, 2 * sizeof(int));
    cudaMemcpy(cuda_capacity, (int*)&capacity, sizeof(int), cudaMemcpyHostToDevice);

    bool *allTrues = (bool *)malloc(compare_size * sizeof(bool));
    memset(allTrues, true, compare_size * sizeof(bool));
    cudaMemcpy(comparisons, allTrues, compare_size * sizeof(bool), cudaMemcpyHostToDevice);

    while (1) {

        cudaMemset(table, 0, table_size * sizeof(char));

        char buffer[INITIAL_CAPACITY];
        printf("Input a sequence to curl:\n");
        scanf("%s", buffer);

        int seqLength = strlen(buffer);
        cudaMemcpy(sequence, buffer, seqLength * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(size, (int*)&seqLength, sizeof(int), cudaMemcpyHostToDevice);

        initializeTable(table, sequence, seqLength);

        int end = capacity - seqLength ;

        for (int i(0); i < end; i++) {
            findCurl(table, sequence, temps, comparisons, size, curl, seqLength);
            seqLength++;
            fillRow << < dim3(seqLength, 1), 1 >> > (table, sequence, size);
        }

        cudaMemcpy(seq, sequence, capacity * sizeof(char), cudaMemcpyDeviceToHost);
        for(int i = 0; i < (capacity - 1); i++) {
            printf("%c ", seq[i]);
        }
        printf("\n\n");
    }


    free(seq);
    cudaFree(table);
    cudaFree(sequence);
    cudaFree(temps);
    cudaFree(comparisons);
    cudaFree(size);
    cudaFree(cuda_capacity);
    cudaFree(curl);


    return 0;
}