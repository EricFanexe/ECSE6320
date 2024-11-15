# Dictionary Codec Project

## Overview

This project implements a dictionary codec to compress data and accelerate search/scan operations, leveraging both multi-threading and SIMD instructions for performance improvement. Dictionary encoding replaces each unique data item with its dictionary ID, which reduces the data footprint and enhances search efficiency. This implementation includes both encoded column generation and querying capabilities, along with a vanilla baseline for performance comparison.

## Requirements and Objectives

### Encoding Functionality
- Given a raw column data file, the program performs dictionary encoding, producing both a dictionary file and an encoded column file.
- Supports multi-threaded dictionary encoding for efficiency.
  
### Query Operations
- **Single Item Search**: Checks if a data item exists in the encoded column. If it exists, returns the indices of all matching entries.
- **Prefix Search**: Given a prefix, searches and returns all unique matching data and their indices.
- The query operations support SIMD instructions to accelerate search/scan speeds.

### Vanilla Column Scan
- Implements a baseline scan of the original data (without dictionary encoding) to compare performance with the dictionary-based implementation.

## Files and Structure

- `main.cpp`: Main program implementing the dictionary codec, including encoding, querying, multi-threading, and SIMD functionality.
- `generate_test.py`: Generates a raw test data file with customizable line count and line length for performance evaluation.
- `run.sh`: Script for compiling and executing the program with different configurations (e.g., varying thread counts). Stores performance results in `results.txt`.

### Generated Files
- `dictionary.txt`: Contains the unique data items and their assigned dictionary IDs.
- `encoded_data.txt`: Contains the encoded data column, with each entry represented by its dictionary ID.
- `results.txt`: Stores the results of performance tests, including encoding and query speeds for various configurations.

## Usage

### Compiling and Running

1. Compile the program:
    ```bash
    ./run.sh
    ```

2. Execute the program with specific parameters:
    ```bash
    ./main <search_value> <prefix> <data_path>
    ```
   - `<search_value>`: The value to be searched in single item search.
   - `<prefix>`: The prefix to be searched in prefix scan.
   - `<data_path>`: Path to the raw column data file.

### Example
```bash
./main apple a ./test.txt
```

### Multi-threading and SIMD
- Set the number of threads by exporting the `NUM_THREADS` environment variable (e.g., `export NUM_THREADS=4`).
- The program uses SIMD instructions (if supported by the system) for accelerated search/scan operations.

## Performance Evaluation

### Encoding Performance

To evaluate encoding performance:
1. Run the program with different thread counts (e.g., 1, 2, 4, 8) to measure encoding time.
2. Record encoding speed with each thread count for analysis and comparison.

### Query Performance

To evaluate query performance:
1. Measure and record the search time and prefix scan time under the following configurations:
   - Vanilla baseline (without dictionary encoding)
   - Dictionary encoding without SIMD
   - Dictionary encoding with SIMD (if supported by the system)

### Experimental Results

**Encoding Performance**:
- Encoding speed with different thread counts: [Fill in with experimental data]

**Single Item Search Speed**:
- Vanilla baseline: [Fill in with experimental data]
- Dictionary (without SIMD): [Fill in with experimental data]
- Dictionary (with SIMD): [Fill in with experimental data]

**Prefix Scan Speed**:
- Vanilla baseline: [Fill in with experimental data]
- Dictionary (without SIMD): [Fill in with experimental data]
- Dictionary (with SIMD): [Fill in with experimental data]

## Analysis and Conclusion

This section summarizes the performance gains achieved through dictionary encoding, multi-threading, and SIMD instructions:

1. **Multi-threading Analysis**: [Discuss the impact of varying thread counts on encoding speed, including any observed speedup or bottlenecks.]

2. **SIMD Utilization**: [Describe the effect of SIMD on query operations. If SIMD significantly improved query performance, highlight this advantage, especially for prefix scans and single item searches.]

3. **Comparison with Vanilla Baseline**: [Compare dictionary encoding performance to the vanilla baseline, explaining how the dictionary codec enhances search/scan efficiency.]

4. **Conclusion and Future Optimizations**: [Summarize the findings and suggest potential future improvements, such as further optimizing multi-threading efficiency, exploring other dictionary management techniques, or refining SIMD operations.]
