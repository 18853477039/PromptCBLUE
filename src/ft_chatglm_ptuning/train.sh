PRE_SEQ_LEN=128
LR=2e-2
#LR=5e-4 # start from 1e-4
your_data_path="/root/PromptCBLUE/datasets/PromptCBLUE/trains/"  # 填入数据集所在的文件夹路径
#your_data_path="/root/PromptCBLUE/datasets/PromptCBLUE/toy_examples/"  # 填入数据集所在的文件夹路径
your_checkpoint_path="/root/autodl-fs/data2/models/ChatGLM"  # 填入用来存储模型的路径
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--THUDM--chatglm-6b/snapshots/bf0f5cfb575eebebf9b655c5861177acfee03f16"   # LLM底座模型路径，或者是huggingface hub上的模型名称
checkpoint="checkpoint-100"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0 和 GPU 1

python src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpoint_path/ChatGLM-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --cache_dir $your_checkpoint_path/ChatGLM-6b-pt-$PRE_SEQ_LEN-$LR \
    --max_train_samples 10000 \
    --fp16 \
    --preprocessing_num_workers 4 \
    --report_to wandb


    # --ptuning_checkpoint $your_checkpoint_path/ChatGLM-6b-pt-$PRE_SEQ_LEN-2e-2/$checkpoint \