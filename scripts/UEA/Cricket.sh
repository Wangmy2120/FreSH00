export CUDA_VISIBLE_DEVICES=0






    python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/Cricket/\
  --model_id Cricket \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --temperature 1 \
  --Seg_num 1 \
  --SegE_num 1  \
  --GE_num 1 \
  --itr 5 \
  --learning_rate 0.0005 \
  --train_epochs 80 \
  --patience 40