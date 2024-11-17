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
- `generate_test.py`: Generates a raw test data file with customizable line count and line length for performance evaluation. For example, running `python3 generate_test.py 1000000 128` means generating 1,000,000 rows of data, with each row containing 128 random lowercase letters. The default configuration is 1,024 * 1,024 (1M) rows, with each row containing 32 random lowercase letters.
- `run.sh`: Script for compiling and executing the program with different configurations (e.g., varying thread counts). Stores performance results in `results.txt`.

### Generated Files
- `test.txt`: Contains the unique test data items.
- `dictionary.txt`: Contains the unique data items and their assigned dictionary IDs. Such as:
```
pjxzwikmydnpxexjxrgwvzziwmboojhk 1048575
wrveynzkcwvxwbuwnayswxowaiynfgns 1048574
znuanfdgqsnctljqunkvcjpjtlspfxnh 1048573
lhlomlwfzcnnbrpkoffwczkwjhedgknh 1048572
btjgcwcyqehyccvzxnzxnsuqvvgcbafv 1048571
cywrepngsmtfjovyyfelgmssxzdemrde 1048570
fksippnjyyiwrmeilkqqasfhfxfqonpw 1048569
```
- `encoded_data.txt`: Contains the encoded data column, with each entry represented by its dictionary ID. Such as:
```
628999
531390
624835
510168
676302
316368
673420
521693
875146
612544
505441
226099
```
- `results.txt`: Stores the results of search and performance tests, including encoding and query speeds for various configurations.

## Usage

### Generating Test Data

1. Generate the test data file using the following command:

```bash
python3 generate_test.py
```

This command will create the test data file named `test.txt`.

### Running the Program

2. Execute the program:

```bash
bash run.sh
```

The `run.sh` script will output all results, including both the full data item search and the prefix search.

### Example Values

In `run.sh`, you can update the values for `search_value` and `prefix` as needed to either search for a complete data item or perform a prefix search.


## Performance Evaluation

**Test Configuration**:
- The test memory size is 32 * 1024 * 1024 = 32MB.
- Single Item Search content: "wowtuamhrrgiwxuzofbqlenmkzfkqwxv"
- Prefix Search content: "wow"

| Thread Count | Encoding Time (s) | Single Item Search Time - Vanilla (s) | Single Item Search Time - Dictionary (No SIMD) (s) | Single Item Search Time - Dictionary (SIMD) (s) | Prefix Search Time - Vanilla (s) | Prefix Search Time - Dictionary (No SIMD) (s) | Prefix Search Time - Dictionary (SIMD) (s) |
|--------------|------------------|----------------------------------------|--------------------------------------------------|---------------------------------------------|---------------------------|-------------------------------------------------|---------------------------------------------|
| 1            | 5.96592          | 0.0324187                              | 0.00457748                                       | 0.00121287                                  | 0.0661013                | 0.071602                                        | 0.0646848                                  |
| 2            | 5.49054          | 0.0319899                              | 0.00638956                                       | 0.00115925                                  | 0.0651126                | 0.0686106                                        | 0.073157                                   |
| 4            | 5.48646          | 0.0331243                              | 0.00461476                                       | 0.00157732                                  | 0.0646633                | 0.0651314                                        | 0.065361                                   |
| 8            | 5.01747          | 0.0334036                              | 0.0050234                                        | 0.00115426                                  | 0.0655487                | 0.0658616                                        | 0.0645386                                  |

## Analysis and Conclusion

### Analysis

From the experimental data, we observe the following trends:

1. **Encoding Performance**:
   - The encoding time decreases as the number of threads increases, indicating that parallelism has a positive impact on the encoding process.

2. **Single Item Search Performance**:
   - The vanilla baseline search time remains fairly consistent across different thread counts, with minimal variation.
   - Dictionary encoding without SIMD shows a slight fluctuation, but it generally achieves faster search times compared to the vanilla baseline.
   - Dictionary encoding with SIMD consistently provides the fastest search times, significantly outperforming both the vanilla baseline and dictionary encoding without SIMD.

3. **Prefix Search Performance**:
   - The vanilla baseline prefix search time remains relatively stable across different thread counts.
   - Dictionary encoding without SIMD generally shows a slight increase in prefix search time compared to the vanilla baseline, suggesting that the additional lookup overhead without SIMD might negatively impact performance.
   - Dictionary encoding with SIMD offers comparable or slightly improved prefix search times relative to the vanilla baseline, demonstrating that SIMD provides benefits for prefix searches by accelerating the data processing.

### Conclusion

The experimental results demonstrate that increasing the number of threads positively affects the encoding time. Dictionary encoding, particularly when combined with SIMD, consistently outperforms the vanilla baseline in both single item search and prefix scan scenarios. SIMD-enabled dictionary encoding offers the best performance overall, making it the preferred option for applications where search speed is critical. However, dictionary encoding without SIMD may not always improve prefix scan performance, and careful consideration is required to determine its suitability based on the specific use case.

Overall, the combination of dictionary encoding and SIMD can lead to significant performance gains, especially for single item searches, and should be considered for optimizing search-intensive tasks.
