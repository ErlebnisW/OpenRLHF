export HF_ENDPOINT="https://hf-mirror.com"

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset PKU-Alignment/Align-Anything-Instruction-100K \
   --input_key prompt \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain /home/wmz/checkpoints/llama3-8b \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 4e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb true \
   --wandb_project Formal-SFT \
   --wandb_run_name llama3-align-anything 
EOF
    
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
