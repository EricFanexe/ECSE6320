#!/bin/bash 
MLC="./mlc" 
RESULTS="./latency_results.csv" 
BUFFER_SIZES="100 400 1000 2000 4000 10000 15000 20000 40000 100000 500000 1000000 3000000 5000000" 

echo "Buffer_size(KB),Latency(ns)" > $RESULTS 
# 循环不同缓冲区大小并记录结果 
for buffer_size in $BUFFER_SIZES; 
do 
    output=$(sudo $MLC --idle_latency -b$buffer_size)
    latency=$(echo "$output" | grep "Each iteration" | awk '{print $9}') 
    echo "$buffer_size,$latency" >> $RESULTS 
done 
echo "Latency results saved to $RESULTS"
