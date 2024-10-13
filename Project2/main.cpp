#include <iostream>
#include <fstream>
#include <chrono>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <atomic>

// Matrix structure
struct Matrix {
    int rows, cols;
    union {
        float *dataFloat;
        int16_t *dataFixed;
    };
    bool isFloat;
};

// Read matrix from txt file
Matrix readMatrix(const char *filename) {
    std::ifstream file(filename);
    Matrix m;
    file >> m.rows >> m.cols >> m.isFloat;
    if (m.isFloat) {
        m.dataFloat = new float[m.rows * m.cols];
        for (int i = 0; i < m.rows * m.cols; i++) {
            file >> m.dataFloat[i];
        }
    } else {
        m.dataFixed = new int16_t[m.rows * m.cols];
        for (int i = 0; i < m.rows * m.cols; i++) {
            file >> m.dataFixed[i];
        }
    }
    return m;
}

// Write matrix to txt file
void writeMatrix(const Matrix &m, const char *filename) {
    std::ofstream file(filename);
    file << m.rows << " " << m.cols << " " << m.isFloat << std::endl;
    if (m.isFloat) {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                file << m.dataFloat[i * m.cols + j] << " ";
            }
            file << std::endl;
        }
    } else {
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                file << m.dataFixed[i * m.cols + j] << " ";
            }
            file << std::endl;
        }
    }
}

// SIMD Matrix multiplication
Matrix mulMatrixIntrin(const Matrix & a,
    const Matrix & b) {
    Matrix c;
    c.rows = a.rows;
    c.cols = b.cols;
    c.isFloat = a.isFloat || b.isFloat;
    if (c.isFloat) {
        c.dataFloat = new float[c.rows * c.cols];
        for (int i = 0; i < c.rows; i++) {
            for (int j = 0; j < c.cols; j++) {
                __m256 sum = _mm256_setzero_ps();
                for (int k = 0; k < a.cols; k += 8) {
                    __m256 aData = _mm256_loadu_ps(a.dataFloat + i * a.cols + k);
                    __m256 bData = _mm256_loadu_ps(b.dataFloat + k * b.cols + j);
                    // sum = _mm256_fmadd_ps(aData, bData, sum);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(aData, bData));
                }
                c.dataFloat[i * c.cols + j] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
            }
        }
    } else {
        c.dataFixed = new int16_t[c.rows * c.cols];
        for (int i = 0; i < c.rows; i++) {
            for (int j = 0; j < c.cols; j++) {
                __m256i sum = _mm256_setzero_si256();
                for (int k = 0; k < a.cols; k += 8) {
                    __m256i aData = _mm256_loadu_si256((__m256i * )(a.dataFixed + i * a.cols + k));
                    __m256i bData = _mm256_loadu_si256((__m256i * )(b.dataFixed + k * b.cols + j));
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(aData, bData));
                }
                int32_t result[8];
                _mm256_storeu_si256((__m256i * ) result, sum);
                c.dataFixed[i * c.cols + j] = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
            }
        }

    }
    return c;
}

void multiThreadedMultiply(const Matrix &a, const Matrix &b, Matrix &c, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < c.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; ++k) {
                sum += a.dataFloat[i * a.cols + k] * b.dataFloat[k * b.cols + j];
            }
            c.dataFloat[i * c.cols + j] = sum;
        }
    }
}

Matrix mulMatrixCommon(Matrix a, Matrix b)
{
    Matrix c;
    c.rows = a.rows;
    c.cols = b.cols;
    c.isFloat = a.isFloat;

    if (a.isFloat)
    {
        c.dataFloat = new float[c.rows * c.cols];
        for (int i = 0; i < c.rows; i++)
        {
            for (int j = 0; j < c.cols; j++)
            {
                float sum = 0;
                for (int k = 0; k < a.cols; k++)
                {
                    sum += a.dataFloat[i * a.cols + k] * b.dataFloat[k * b.cols + j];
                }
                c.dataFloat[i * c.cols + j] = sum;
            }
        }
    }
    else
    {
        c.dataFixed = new int16_t[c.rows * c.cols];
        for (int i = 0; i < c.rows; i++)
        {
            for (int j = 0; j < c.cols; j++)
            {
                int sum = 0;
                for (int k = 0; k < a.cols; k++)
                {
                    sum += (int)(a.dataFixed[i * a.cols + k] * b.dataFixed[k * b.cols + j]) >> 8;
                }
                c.dataFixed[i * c.cols + j] = sum;
            }
        }
    }

    return c;
}

// Function for block matrix multiplication to improve cache locality
void blockMatrixMultiply(const Matrix &a, const Matrix &b, Matrix &c, int blockSize) {
    for (int iBlock = 0; iBlock < c.rows; iBlock += blockSize) {
        for (int jBlock = 0; jBlock < c.cols; jBlock += blockSize) {
            for (int kBlock = 0; kBlock < a.cols; kBlock += blockSize) {
                // Multiply blocks
                for (int i = iBlock; i < std::min(iBlock + blockSize, c.rows); i++) {
                    for (int j = jBlock; j < std::min(jBlock + blockSize, c.cols); j++) {
                        float sum = 0.0f;
                        for (int k = kBlock; k < std::min(kBlock + blockSize, a.cols); k++) {
                            sum += a.dataFloat[i * a.cols + k] * b.dataFloat[k * b.cols + j];
                        }
                        c.dataFloat[i * c.cols + j] += sum;
                    }
                }
            }
        }
    }
}

// SIMD + Cache Optimization + Threaded Matrix multiplication
void blockMatrixMultiplySIMD(const Matrix &a, const Matrix &b, Matrix &c, int startRow, int endRow, int blockSize) {
    for (int iBlock = startRow; iBlock < endRow; iBlock += blockSize) {
        for (int jBlock = 0; jBlock < c.cols; jBlock += blockSize) {
            for (int kBlock = 0; kBlock < a.cols; kBlock += blockSize) {
                // Multiply blocks
                for (int i = iBlock; i < std::min(iBlock + blockSize, endRow); i++) {
                    for (int j = jBlock; j < std::min(jBlock + blockSize, c.cols); j++) {
                        __m256 sum = _mm256_setzero_ps();
                        for (int k = kBlock; k < std::min(kBlock + blockSize, a.cols); k += 8) {
                            __m256 aData = _mm256_loadu_ps(a.dataFloat + i * a.cols + k);
                            __m256 bData = _mm256_loadu_ps(b.dataFloat + k * b.cols + j);
                            sum = _mm256_add_ps(sum, _mm256_mul_ps(aData, bData));
                        }
                        c.dataFloat[i * c.cols + j] += sum[0] + sum[1] + sum[2] + sum[3] +
                                                        sum[4] + sum[5] + sum[6] + sum[7];
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrixA file> <matrixB file> <method(s)>" << std::endl;
        std::cerr << "Method options: intrin, common, multithread, or combinations like multithread+intrin" << std::endl;
        return 1;
    }

    Matrix a = readMatrix(argv[1]);
    Matrix b = readMatrix(argv[2]);

    // Check if matrices can be multiplied
    if (a.cols != b.rows) {
        std::cerr << "Matrix dimensions are not compatible for multiplication." << std::endl;
        return 1;
    }

    // Initialize result matrix
    Matrix c;
    c.rows = a.rows;
    c.cols = b.cols;
    c.isFloat = a.isFloat;
    c.dataFloat = new float[c.rows * c.cols];

    std::string method(argv[3]);

    auto start = std::chrono::high_resolution_clock::now();

    if (method == "intrin") {
        c = mulMatrixIntrin(a, b);
    } else if (method == "common") {
        c = mulMatrixCommon(a, b);
    } else if (method == "multithread") {
        const int numThreads = std::thread::hardware_concurrency();
        // const int numThreads = std::min(static_cast<unsigned int>(2), std::thread::hardware_concurrency()); // 限制为最多2个线程
        // std::vector<std::thread> threads;
        // int rowsPerThread = c.rows / numThreads;
        // for (int i = 0; i < numThreads; ++i) {
        //     int startRow = i * rowsPerThread;
        //     int endRow = (i == numThreads - 1) ? c.rows : startRow + rowsPerThread;
        //     threads.emplace_back(multiThreadedMultiply, std::ref(a), std::ref(b), std::ref(c), startRow, endRow, 1000);
        // }
        // for (auto &t : threads) t.join();

        std::vector<std::thread> threads;

        // Split the work among threads
        int rowsPerThread = c.rows / numThreads;
        int remainingRows = c.rows % numThreads;
        int currentRow = 0;

        for (int i = 0; i < numThreads; ++i) {
            int numRows = rowsPerThread + (i < remainingRows ? 1 : 0);
            threads.emplace_back(multiThreadedMultiply, std::ref(a), std::ref(b), std::ref(c), currentRow, currentRow + numRows);
            currentRow += numRows;
        }

        // Wait for all threads to finish
        for (auto& thread : threads) thread.join();
    } else if (method == "block") {
        int blockSize = 64; // Block size based on cache line size and system L1 cache size
        blockMatrixMultiply(a, b, c, blockSize);
    } else if(method == "multithread+intrin+block") {
        const int numThreads = std::thread::hardware_concurrency();
        // const int numThreads = std::min(static_cast<unsigned int>(2), std::thread::hardware_concurrency()); // 限制为最多2个线程
        std::vector<std::thread> threads;
        int blockSize = 64; // Block size to improve cache locality
        int rowsPerThread = c.rows / numThreads;

        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? c.rows : startRow + rowsPerThread;
            threads.emplace_back(blockMatrixMultiplySIMD, std::ref(a), std::ref(b), std::ref(c), startRow, endRow, blockSize);
        }

        for (auto &t : threads) t.join();
    } else {
        std::cerr << "Invalid method specified." << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Duration: " << duration << " milliseconds" << std::endl;

    if (c.isFloat) {
    delete[] c.dataFloat;
} else {
    delete[] c.dataFixed;
}
    return 0;
}
