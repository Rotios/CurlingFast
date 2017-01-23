//(c) 2017 John Freeman and Jose Rivas

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 1024
#define X(r, c) ((r+1) * (r * 0.5)) + c  

void printTable(char *table, int length) {
    char *CPUTable;

    CPUTable = (char *)malloc(length * sizeof(char));
    cudaMemcpy(CPUTable, table, length * sizeof(char), cudaMemcpyDeviceToHost);

    for (int i(0); i < length; ++i) {
        printf("%c ", CPUTable[i] );
    }
    
    printf("\n");
    free(CPUTable);
}

void printBoolArray(bool *comparisons, int size){
    
    bool *comp;
    comp = (bool *)malloc(size * sizeof(bool));

    cudaMemcpy(comp, comparisons, size * sizeof(bool), cudaMemcpyDeviceToHost);
    printf("comparisons =  ");
    for(int i(0); i < size; ++i){
        printf("%d ", comp[i]);
    }

    printf("\n");
}

// Magic if it works
__global__ void fillRow(char *table, char *sequence, int *index) {
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
        fillRow << < dim3(i + 1, 1), 1 >> > (table, sequence, index);
    }

    cudaFree(index);
}

__global__ void cudaMin(char *table, char *temps, bool *comparisons, int *size) {
    int colIdx = blockIdx.y;

    int rowIdx = blockIdx.x;

    int boolId = colIdx * (*size) + rowIdx;
    int idx = X(*size - 1 - rowIdx, blockIdx.y);

    if (comparisons[boolId]) {
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
    int tabIdx = X((tabRowIdx >= colIdx) * tabRowIdx, colIdx);
    int tabIdy = X((tabRowIdx >= colIdx) * tabRowIdy, colIdx);

    //temps[colIdx] = (char)tabRowIdx;
    //temps[colIdx + seqLength] = (char)tabRowIdy;

    int boolId = colIdx * seqLength + rowIdx;

    int xval = (table[tabIdx] - '0');
    int yval = (table[tabIdy] - '0');

    if (yval == 0 || yval == xval) {}
    else if (xval == 0 || xval > yval) {
        comparisons[boolId] = false;
    }
}

void findCurl(char *table, char *sequence, char *temps, bool *comparisons, int *size, int *curl, int seqLength) {

    int numBlocks = seqLength >> 1;
    minCompare << < numBlocks, dim3(numBlocks, numBlocks) >> > (table, temps, comparisons, size);
    printBoolArray(comparisons, 1000);
    cudaMin << < dim3(numBlocks, numBlocks), 1 >> > (table, temps, comparisons, size);

    char *mins;
    mins = (char *)malloc(1000 * sizeof(char));

    cudaMemcpy(mins, temps, seqLength * sizeof(char), cudaMemcpyDeviceToHost);
    printf("temps =  ");
    for(int i(0); i < seqLength * seqLength; ++i){
        printf("%d ", mins[i]);
    }

    printf("\n");

    //numBlocks = (numBlocks * numBlocks) * 0.5f;
    /*maxCompare << < dim3(numBlocks, numBlocks), numBlocks >> > (temps, comparisons, size);
    cudaMax << < dim3(numBlocks, numBlocks), 1 >> > (sequence, comparisons, size, curl);*/

    /*if (seqLength < INITIAL_CAPACITY) {
        int len = *size++;
        sequence[len] = *curl;
    }*/
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
        printf("%d\n\n", seqLength);
        printTable(table, (seqLength * (seqLength + 1))/2);

        for (int i(0); i < 1; i++) {
            findCurl(table, sequence, temps, comparisons, size, curl, seqLength);
            //fillRow << < dim3(seqLength++, 1), 1 >> > (table, sequence, size);
        }
        
        printf("\n\n");
    }


    cudaFree(table);
    cudaFree(sequence);
    cudaFree(temps);
    cudaFree(comparisons);
    cudaFree(size);
    cudaFree(cuda_capacity);
    cudaFree(curl);


    return 0;
}


/*
        char *seq = (char *) malloc(INITIAL_CAPACITY*sizeof(char));

        cudaMemcpy(seq, (int*)&sequence, seqLength * sizeof(char), cudaMemcpyDeviceToHost);
        for (int i = 0; i < seqLength; ++i) {
            printf("%c ", seq[i]);
        }*/
