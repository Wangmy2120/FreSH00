
  export CUDA_VISIBLE_DEVICES=0
    python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/AtrialFibrillation/\
  --model_id AtrialFibrillation \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --MoE_flag 4 \
  --temperature 1 \
  --Seg_num 1 \
  --SegE_num 1  \
  --GE_num 1 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 300 \
  --patience 50