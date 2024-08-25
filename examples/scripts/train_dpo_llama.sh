export HF_ENDPOINT="https://hf-mirror.com"

set -x

project_name=DPO-MIX-7k
run_name=llama3-mix-dpo-0.1

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/${run_name} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain /data/wmz_workspace/checkpoints/llama3-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --adam_offload \
   --learning_rate 5e-7 \
   --beta 1.1 \
   --dataset argilla/dpo-mix-7k \
   --apply_chat_template \
   --train_split train \
   --eval_split eval \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb True \
   --wandb_run_name ${run_name}
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
