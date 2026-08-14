export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/\
  --model_id Heartbeat \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 20 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 2 \
  --GE_num 2 \
  --temperature 3.3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.003 \
  --train_epochs 200 \
  --patience 50