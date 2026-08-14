export CUDA_VISIBLE_DEVICES=0


 python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PEMS-SF/\
  --model_id PEMS-SF \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 3 \
  --GE_num 3 \
  --temperature 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.005 \
  --train_epochs 60 \
  --patience 30