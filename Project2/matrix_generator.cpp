#include <fstream>
#include <iostream>
#include <random>
#include <string>

// Declare random number generator at file scope
std::random_device rd;
std::mt19937 rng(rd()); // Unified random generator for both float and int16

// Function to generate a random float between -1.0 and 1.0
float randomFloat() {
    static std::uniform_real_distribution<float> dist(-1.0, 1.0);
    return dist(rng);
}

// Function to generate a random 2-byte fixed-point value between -32767 and 32767
int16_t randomFixedPoint() {
    static std::uniform_int_distribution<int16_t> dist(-32767, 32767);
    return dist(rng);
}

// Function to generate and save a dense or sparse matrix to a file
void generateMatrix(int rows, int cols, bool isFloat, const std::string &filename, float sparsity = 0.0f) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: Unable to open file '" << filename << "' for writing." << std::endl;
        exit(1); // Exit if the file cannot be written
    }

    // Write matrix dimensions and data type to the file
    ofs << rows << " " << cols << " " << (isFloat ? "float" : "int16") << std::endl;

    // Generate and write random values to the file
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // For sparse matrices, randomly set some values to zero based on sparsity
            if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < sparsity) {
                ofs << 0 << " ";  // Zero for sparsity
            } else {
                if (isFloat) {
                    ofs << randomFloat() << " ";
                } else {
                    ofs << randomFixedPoint() << " ";
                }
            }
        }
        ofs << std::endl;
    }

    std::cout << "Matrix saved to '" << filename << "' with sparsity: " << sparsity << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dimension> <isFloat (1/0)> <sparsity>" << std::endl;
        return 1;
    }

    int dimension = std::stoi(argv[1]);
    bool isFloat = std::stoi(argv[2]) != 0;
    float sparsity = std::stof(argv[3]);

    std::string filenameA = "matrixA_" + std::to_string(dimension) + "_" + (isFloat ? "float" : "int16") + ".txt";
    std::string filenameB = "matrixB_" + std::to_string(dimension) + "_" + (isFloat ? "float" : "int16") + ".txt";

    // Generate matrices A and B with the specified sparsity
    generateMatrix(dimension, dimension, isFloat, filenameA, sparsity);
    generateMatrix(dimension, dimension, isFloat, filenameB, sparsity);

    return 0;
}
