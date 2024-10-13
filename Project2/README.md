## Matrix-Matrix Multiplication
### Instructions
In this project, we need to implement a C/C++ module that carries out high-speed matrix-matrix multiplication by explicitly utilizing.

(i) Multiple threads

(ii) x86 SIMD instructions

(iii) Techniques to minimize cache miss rate via restructuring data access patterns (as discussed in class)

The implementation should be able to support (1) configurable matrix size that can be much larger than the on-chip cache capacity, and (2) both fixed-point and floating-point data. Moreover, your implement should allow users to individually turn on/off the three optimization techniques (i.e., multi-threading, SIMD, and cache miss minimization) and configure the thread number so that users could easily observe the effect of any combination of these three optimization techniques.

### Quick Start
In this project, there are two program, the first one named ```matrix_generator.cpp```, which aims to generate random matrix A and B. To use it, we need firstly compile it by using:
```
g++ -o matrix_generator matrix_generator.cpp
```
After get the exe file named ```matrix_generator```, then use ```./matrix_generator <dimension> <isFloat> <sparsity>```. For example 1000 dimension float32 matrix with 1% sparsity:
```
./matrix_generator 1000 1 0.01
```
Which will generate two matrix A: ```matrixA_1000_float.txt``` and matrix B: ```matrixB_1000_float.txt ```

The other program is ```main.cpp```, which will execute the matrix-matrix multiplication in four modes: 1)Multiple threads, 2)x86 SIMD instructions, 3)Techniques to minimize cache miss rate via restructuring data access patterns, and 4)Simultaneous apply 1) 2) 3).

Firstly, compile it by command line:
```
g++ -march=native -o main main.cpp -std=c++11 -pthread
```
Then to execute the multiplication, use ```./main <matrixA> <matrixB> <mode>```, for example:
```
./main matrixA_1000_float.txt matrixB_1000_float.txt multithread+intrin+block
./main matrixA_1000_float.txt matrixB_1000_float.txt common
./main matrixA_1000_float.txt matrixB_1000_float.txt multithread
./main matrixA_1000_float.txt matrixB_1000_float.txt block
./main matrixA_1000_float.txt matrixB_1000_float.txt intrin
```
The output will be the execution time of any modes, like:
```
Duration: 622 milliseconds
```

When using block mode, you can adjust the block size by modifying the following statement in ```main.cpp```:
```cpp
int blockSize = 64; // Block size to improve cache locality
```

When using multithread mode, you can adjust the number of threads by modifying the following statement in ```main.cpp```:
```cpp
const int numThreads = std::min(static_cast<unsigned int>(2), std::thread::hardware_concurrency());
```
You can change `static_cast<unsigned int>(2)` to set the desired number of threads. Alternatively, you can automatically determine the number of threads based on the system's hardware by using the following statement:
```cpp
const int numThreads = std::thread::hardware_concurrency();
```

### Evaluation
Initially, I show the performance of my code under different matrix configuration: 
1) Float32 1,000x1,000 with 0.1% sparsity matrix
2) Float32 1,000x1,000 with 1% sparsity matrix
3) Int16 5,000x5,000 with 30% sparsity matrix
4) Int16 10,000x10,000 with 1% sparsity matrix

Float32 1,000x1,000 with 0.1% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|3860|621|1116|7342|223|

Float32 1,000x1,000 with 1% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|3817|639|1115|7237|220|
   
Int16 5,000x5,000 with 30% sparsity matrix results(milliseconds), and due to the large size of the matrix, the number of threads is limited to 2

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|428369|224247|86297|986426|74755|

Int16 10,000x10,000 with 1% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|4045202|too large|855851|too slow|too large|

Matrix Multiplication with Different Sparsity Levels:

Two Float32 1,000x1,000 with 1% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|3817|639|1115|7237|220|

Two Float32 1,000x1,000 with 95% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|3824|601|1113|7299|222|

One Float32 1,000x1,000 with 95% sparsity matrix and one Float32 1,000x1,000 with 1% sparsity matrix results(milliseconds)

| **Original** | **Multi Threads** | **SIMD** | **Block 64** | **Multi Threads+SIMD+Block 64** |
|-------------|----------------------------|---------------|--------------------|----------------------------------------------|
|3817|622|1116|7299|235|

### Conclusion
The performance evaluation shows significant improvements with optimization techniques for matrix-matrix multiplication:

1) Multi-threading consistently improves speed, especially for large matrices, but its benefit reduces as matrix size increases due to hardware limitations.

2) SIMD offers noticeable gains, particularly for dense matrices, but is less impactful than multi-threading.

3) Block Matrix Optimization helps reduce cache misses, but for small matrices, it performs poorly due to overhead. For larger matrices, its effectiveness improves but still lags behind SIMD and multi-threading.

4) Combining Multi-threading, SIMD, and Block Optimization provides the best performance across all matrix sizes and configurations, with up to 20x speedup compared to the original method.

5) Sparse matrices benefit most from SIMD and multi-threading. For highly sparse matrices, all optimization methods show minimal differences, as memory access becomes the main bottleneck.

6) Extremely large matrices, such as 10,000x10,000, challenge system resources, making combined optimizations impractical at such scales.

In summary, combining all three optimizations—multi-threading, SIMD, and block matrix processing—delivers the best performance across various matrix sizes and sparsity levels.
