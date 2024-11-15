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
./run.sh
```

The `run.sh` script will output all results, including both the full data item search and the prefix search.

### Example Values

In `run.sh`, you can update the values for `search_value` and `prefix` as needed to either search for a complete data item or perform a prefix search.


## Performance Evaluation

**Test Configuration**:
- The test memory size is 32 * 1024 * 1024 = 32MB.
- Single Item Search content: "wowtuamhrrgiwxuzofbqlenmkzfkqwxv"
- Prefix Search content: "wow"

| Thread Count | Vanilla Single Item Search Time (s) | Dictionary Single Item Search Time (No SIMD) (s) | Dictionary Single Item Search Time (SIMD) (s) | Vanilla Prefix Search Time (s) | Dictionary Prefix Search Time (No SIMD) (s) | Dictionary Prefix Search Time (SIMD) (s) |
|--------------|-------------------------------------|------------------------------------------------|----------------------------------------------|-------------------------------|---------------------------------------------|-------------------------------------------|
| 1            | 0.0158985                           | 0.00223477                                     | 0.000583437                                  | 0.0323419                    | 0.0322951                                  | 0.033759                                  |
| 2            | 0.0168299                           | 0.00225779                                     | 0.000597504                                  | 0.0343649                    | 0.0325781                                  | 0.0321756                                 |
| 4            | 0.0168108                           | 0.00233418                                     | 0.000573298                                  | 0.0336547                    | 0.0324437                                  | 0.0349848                                 |
| 8            | 0.0165489                           | 0.00225471                                     | 0.000655496                                  | 0.0325406                    | 0.0345408                                  | 0.0340513                                 |


## Analysis and Conclusion

This section summarizes the performance gains achieved through dictionary encoding, multi-threading, and SIMD instructions:

1. **Multi-threading Analysis**: [Discuss the impact of varying thread counts on encoding speed, including any observed speedup or bottlenecks.]

2. **SIMD Utilization**: [Describe the effect of SIMD on query operations. If SIMD significantly improved query performance, highlight this advantage, especially for prefix scans and single item searches.]

3. **Comparison with Vanilla Baseline**: [Compare dictionary encoding performance to the vanilla baseline, explaining how the dictionary codec enhances search/scan efficiency.]

4. **Conclusion and Future Optimizations**: [Summarize the findings and suggest potential future improvements, such as further optimizing multi-threading efficiency, exploring other dictionary management techniques, or refining SIMD operations.]
