  export CUDA_VISIBLE_DEVICES=0

  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/\
  --model_id UWaveGestureLibrary \
  --model FreSH \
  --data UEA \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --Seg_num 4 \
  --SegE_num 2 \
  --GE_num 2 \
  --des 'Exp' \
  --itr 5 \
  --temperature 3.3 \
  --learning_rate 0.002 \
  --train_epochs 100 \
  --patience 30