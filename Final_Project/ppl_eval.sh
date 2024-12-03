MODELPATH=lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/ppl_eval/longchat
mkdir -p $OUTPUT_DIR

budget=64
start_idx=0
min_tokens=512

CUDA_VISIBLE_DEVICES=8,9 python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --start_idx $start_idx \
    --min_tokens $min_tokens \
    --num_eval_tokens 512 \
    --quest --token_budget $budget --chunk_size 16 
