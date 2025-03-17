PRE_SEQ_LEN=128
CHECKPOINT="/root/autodl-fs/data2/models/PromptCBLUE/PromptCBLUE-chatglm-6b-pt-128-2e-2"   # 填入用来存储模型的文件夹路径
STEP=10    # 用来评估的模型checkpoint是训练了多少步

your_data_path="./datasets/PromptCBLUE/toy_examples/"  # 填入数据集所在的文件夹路径
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--THUDM--visualglm-6b/snapshots/a05d5f967eb67d2503fb8ccab6b5e01713305184"   # LLM底座模型路径，或者是huggingface hub上的模型名称


CUDA_VISIBLE_DEVICES=0 python src/ft_chatglm_ptuning/main.py \
    --do_predict \
    --do_eval \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/test.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --ptuning_checkpoint $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN
