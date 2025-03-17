PRE_SEQ_LEN=128
LR=2e-2
#your_data_path="/root/PromptCBLUE/datasets/PromptCBLUE/trains/"  # 填入数据集所在的文件夹路径
your_data_path="/root/PromptCBLUE/datasets/PromptCBLUE/toy_examples/"  # 填入数据集所在的文件夹路径
your_checkpoint_path="/root/autodl-fs/data2/models/PromptCBLUE/"  # 填入用来存储模型的路径
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--THUDM--visualglm-6b/snapshots/a05d5f967eb67d2503fb8ccab6b5e01713305184"   # LLM底座模型路径，或者是huggingface hub上的模型名称


CUDA_VISIBLE_DEVICES=0 python src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpoint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 80 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16



