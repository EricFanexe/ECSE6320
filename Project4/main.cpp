#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <shared_mutex>
#ifdef __AVX2__
#include <immintrin.h>
#endif

using namespace std;

int get_num_threads_from_env() {
    const char* num_threads_str = getenv("NUM_THREADS");
    return num_threads_str ? stoi(num_threads_str) : 4;
}

const int NUM_THREADS = get_num_threads_from_env();

shared_mutex dict_mutex;

// Merge local dictionaries into a global dictionary
void merge_dictionaries(const vector<unordered_map<string, int>>& local_dicts, unordered_map<string, int>& global_dict) {
    for (const auto& local_dict : local_dicts) {
        for (const auto& pair : local_dict) {
            unique_lock lock(dict_mutex); // Reduce lock contention by using shared locks
            if (global_dict.find(pair.first) == global_dict.end()) {
                global_dict[pair.first] = global_dict.size();
            }
        }
    }
}

// Dictionary encoder
void encoder(const vector<string>& data, int start, int end, unordered_map<string, int>& local_dict) {
    for (int i = start; i < end; i++) {
        if (local_dict.find(data[i]) == local_dict.end()) {
            local_dict[data[i]] = local_dict.size();
        }
    }
}

vector<int> encode(const vector<string>& data, unordered_map<string, int>& dict) {
    vector<int> encoded_data(data.size());
    int data_size = data.size();
    int chunk_size = data_size / NUM_THREADS;

    // Local dictionaries for each thread
    vector<unordered_map<string, int>> local_dicts(NUM_THREADS);
    vector<thread> threads;

    for (int i = 0; i < NUM_THREADS; i++) {
        int start = i * chunk_size;
        int end = (i == NUM_THREADS - 1) ? data_size : (i + 1) * chunk_size;
        threads.push_back(thread(encoder, ref(data), start, end, ref(local_dicts[i])));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Merge local dictionaries into the global dictionary
    merge_dictionaries(local_dicts, dict);

    // Encode data using the global dictionary
    for (size_t i = 0; i < data.size(); i++) {
        encoded_data[i] = dict[data[i]];
    }

    // Write dictionary to file
    ofstream dict_file("dictionary.txt");
    for (const auto& pair : dict) {
        dict_file << pair.first << " " << pair.second << endl;
    }
    dict_file.close();

    // Write encoded data to file
    ofstream encoded_file("encoded_data.txt");
    for (int id : encoded_data) {
        encoded_file << id << endl;
    }
    encoded_file.close();

    return encoded_data;
}

// Vanilla Column Scan (no dictionary encoding)
vector<int> vanilla_scan(const vector<string>& data, const string& value) {
    vector<int> indices;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] == value) {
            indices.push_back(i);
        }
    }
    return indices;
}

// Dictionary search for a single item
vector<int> search_encoded(const vector<int>& encoded_data, int value) {
    vector<int> indices;
    for (size_t i = 0; i < encoded_data.size(); i++) {
        if (encoded_data[i] == value) {
            indices.push_back(i);
        }
    }
    return indices;
}

#ifdef __AVX2__
// Dictionary search using SIMD for a single item
vector<int> search_encoded_simd(const vector<int>& encoded_data, int value) {
    vector<int> indices;
    __m256i value_data = _mm256_set1_epi32(value);

    for (size_t i = 0; i < encoded_data.size(); i += 8) {
        __m256i data_i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&encoded_data[i]));
        __m256i cmp = _mm256_cmpeq_epi32(value_data, data_i);

        if (_mm256_movemask_epi8(cmp) != 0) {
            for (int j = 0; j < 8 && (i + j) < encoded_data.size(); ++j) {
                if (encoded_data[i + j] == value) {
                    indices.push_back(i + j);
                }
            }
        }
    }
    return indices;
}

// Prefix search using SIMD for dictionary encoded data
vector<pair<int, int>> encoded_prefix_search_simd(const vector<string>& data, const vector<int>& encoded_data, const unordered_map<string, int>& dict, const string& prefix) {
    vector<pair<int, int>> result;
    unordered_set<int> unique_matches;
    
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i].substr(0, prefix.size()) == prefix) {
            int encoded_val = encoded_data[i];
            if (unique_matches.find(encoded_val) == unique_matches.end()) {
                unique_matches.insert(encoded_val);
                result.emplace_back(encoded_val, i);
            }
        }
    }
    return result;
}
#endif

// Prefix search in vanilla (no dictionary encoding)
vector<pair<string, int>> vanilla_prefix_search(const vector<string>& data, const string& prefix) {
    vector<pair<string, int>> result;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i].substr(0, prefix.size()) == prefix) {
            result.emplace_back(data[i], i);
        }
    }
    return result;
}

// Prefix search in encoded data without SIMD
vector<pair<int, int>> encoded_prefix_search(const vector<string>& data, const vector<int>& encoded_data, const unordered_map<string, int>& dict, const string& prefix) {
    vector<pair<int, int>> result;
    unordered_set<int> unique_matches;

    for (size_t i = 0; i < data.size(); i++) {
        if (data[i].substr(0, prefix.size()) == prefix) {
            int encoded_val = encoded_data[i];
            if (unique_matches.find(encoded_val) == unique_matches.end()) {
                unique_matches.insert(encoded_val);
                result.emplace_back(encoded_val, i);
            }
        }
    }
    return result;
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <search_value> <prefix> <data_path>" << endl;
        return 1;
    }

    // Load data from file
    ifstream infile(argv[3]);
    vector<string> data;
    string line;
    while (getline(infile, line)) {
        data.push_back(line);
    }
    infile.close();

    // Baseline Vanilla Column Scan for single item search
    auto start = chrono::high_resolution_clock::now();
    vector<int> vanilla_result = vanilla_scan(data, argv[1]);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> vanilla_search_time = end - start;

    // Baseline Vanilla Column Scan for prefix search
    start = chrono::high_resolution_clock::now();
    vector<pair<string, int>> vanilla_prefix_result = vanilla_prefix_search(data, argv[2]);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> vanilla_prefix_time = end - start;

    // Dictionary Encoding
    unordered_map<string, int> dict;
    start = chrono::high_resolution_clock::now();
    vector<int> encoded_data = encode(data, dict);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> encoding_time = end - start;


    // Convert search value to encoded format
    string search_value = argv[1];
    int encoded_search_value = dict.count(search_value) ? dict[search_value] : -1;

    if (encoded_search_value == -1) {
        cout << "Search value not found in dictionary." << endl;
        return 1;
    }

    // Dictionary search (without SIMD) for single item
    start = chrono::high_resolution_clock::now();
    vector<int> search_result = search_encoded(encoded_data, encoded_search_value);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dict_search_time = end - start;

    #ifdef __AVX2__
    // Dictionary search using SIMD
    start = chrono::high_resolution_clock::now();
    vector<int> search_simd_result = search_encoded_simd(encoded_data, encoded_search_value);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dict_search_simd_time = end - start;
    #endif

    // Prefix search in dictionary without SIMD
    start = chrono::high_resolution_clock::now();
    vector<pair<int, int>> encoded_prefix_result = encoded_prefix_search(data, encoded_data, dict, argv[2]);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dict_prefix_time = end - start;

    #ifdef __AVX2__
    // Prefix search in dictionary using SIMD
    start = chrono::high_resolution_clock::now();
    vector<pair<int, int>> encoded_prefix_result_simd = encoded_prefix_search_simd(data, encoded_data, dict, argv[2]);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> dict_prefix_simd_time = end - start;
    #endif

    // Print results
    cout << "Encoding time: " << encoding_time.count() << " seconds" << endl;
    cout << "Vanilla single item search time: " << vanilla_search_time.count() << " seconds" << endl;
    cout << "Dictionary single item search time (no SIMD): " << dict_search_time.count() << " seconds" << endl;
    #ifdef __AVX2__
    cout << "Dictionary single item search time (SIMD): " << dict_search_simd_time.count() << " seconds" << endl;
    #endif

    cout << "Vanilla prefix search time: " << vanilla_prefix_time.count() << " seconds" << endl;
    cout << "Dictionary prefix search time (no SIMD): " << dict_prefix_time.count() << " seconds" << endl;
    #ifdef __AVX2__
    cout << "Dictionary prefix search time (SIMD): " << dict_prefix_simd_time.count() << " seconds" << endl;
    #endif

    cout << "\nSingle item search indices in vanilla data: ";
    for (int idx : vanilla_result) cout << idx << " ";
    cout << "\nSingle item search indices in encoded data: ";
    for (int idx : search_result) cout << idx << " ";

    cout << "\nPrefix search unique matches and indices in vanilla data:\n";
    for (const auto& pair : vanilla_prefix_result) cout << pair.first << " at index " << pair.second << endl;
    
    cout << "\nPrefix search unique matches and indices in encoded data:\n";
    for (const auto& pair : encoded_prefix_result) cout << "Encoded Value: " << pair.first << " at index " << pair.second << endl;

    return 0;
}
