#!/bin/bash

g++ -march=native -o main main.cpp -pthread -mavx2

thread_counts="1 2 4 8"
data_path="./test.txt"

rm -f results.txt

for thread_count in $thread_counts; do
    export NUM_THREADS=$thread_count
    echo "Running experiment with $thread_count threads" | tee -a results.txt

    search_value="wowtuamhrrgiwxuzofbqlenmkzfkqwxv"  
    prefix="wow"       

    ./main $search_value $prefix $data_path | tee -a results.txt

    mv dictionary.txt "dictionary_${thread_count}_threads.txt"
    mv encoded_data.txt "encoded_data_${thread_count}_threads.txt"
    echo "Saved dictionary and encoded data for ${thread_count} threads." | tee -a results.txt
    echo -e "\n" | tee -a results.txt
done

echo "Experiments completed. Results saved to results.txt."
