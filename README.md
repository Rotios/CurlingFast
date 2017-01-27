#CUDA Implementation of a Curling Number Solver

(c) 2017 John Freeman and Jose Rivas

<b>Use:</b>
<br>
This program requires CUDA 2.0. When run, it will read in a string representation of an integer sequence. It will then calculate and append the next 32 - N numbers in the curling sequence for the given initial sequence, where N is the size of the initial sequence.

<b>The Implementation:</b>
<br>
Our current implementation for the Curling Number Conjecture takes a dynamic programing approach. It calculates the curl of a given sequence using a flattened triangular table where values can be accessed using the X macro defined at the top of the program. This flattened triangular array represents the bottom portion of a larger 2-dimensional array, where each row represents the indices of our current sequence and each column represents how often the number at each index has occurred previously. We also include a min and max function to perform on the table as required by the algorithm.

<b>The Algorithm:</b>
<br>
When the initial sequence is entered, the table is filled in row-by-row using the X macro. For each column C in row R, we check whether the number at index R in the initial sequence occurred at index R - C in our sequence. If it did, we set position X(R, C) in our table equal to 1 plus the value at position X(R - C, C). If X(R-C, C) falls outside of the bounds of our triangular array, we put a 2 in that position. We then store the minimum values of the last half of each of the rows in the bottom half of the table and return the maximum value of those values as the curling number for that sequence. We then add that number to the end of our sequence, fill in the new row according to that number, and run the algorithm again an arbitrary number of times.

<b>Limitations:</b>
<ul>
<li>
Currently, we have a bug that causes the starting sequence we would like to curl become non-deterministic. We believe that this is a bug involving the cudaMemcpy from the input sequence into the GPU memory.
</li>
<li>
CUDA supports a total of 1024 threads per block. Due to this limitation, our current implementation for the Curling Number Conjecture can only solve for sequences up to 32 characters long, because we are calculating the minimum values for each row all at once. This limitation could be fixed by either finding the minimum in each row either one at a time or in batches so that the computation would only require blocks instead of threads since CUDA allows for many more blocks than threads.
</li>
</ul>
<b>Acknowledgements:</b>
<br>
We started this project in order to figure out if we could get his original implementation to solve for general curling sequences faster through CUDA. While the results were mixed, there seems to be some potential. Thank you, Duane, for your support on this project.
