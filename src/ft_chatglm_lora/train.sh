lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--THUDM--ChatGLM-6B/snapshots/bf0f5cfb575eebebf9b655c5861177acfee03f16"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/PromptCBLUE/trains"  # 填入数据集所在的文件夹路径
your_checkpoint_path="/root/autodl-fs/data2/models/ChatGLM/"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径
resume_from_checkpoint="/root/autodl-fs/data2/models/ChatGLM/PromptCBLUE-ChatGLM-6b-lora-2e-4/checkpoint-3500"  # 如果之前训练过，且存储了模型权重，则设置为模型权重的文件夹路径

CUDA_VISIBLE_DEVICES=0 python src/ft_chatglm_lora/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --cache_dir $your_checkpoint_path/PromptCBLUE-ChatGLM-6b-lora-$PRE_SEQ_LEN-$LR \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpoint_path/PromptCBLUE-ChatGLM-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 828 \
    --max_target_length 196 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --preprocessing_num_workers 12 \
    --report_to wandb \
    --resume_from_checkpoint $resume_from_checkpoint

    # --max_train_samples 5000 \


