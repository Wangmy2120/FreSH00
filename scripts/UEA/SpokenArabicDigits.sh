export CUDA_VISIBLE_DEVICES=0



python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model FreSH \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 3 \
  --GE_num 3 \
  --temperature 3.3 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 15