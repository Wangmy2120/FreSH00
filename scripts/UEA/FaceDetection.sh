export CUDA_VISIBLE_DEVICES=0

  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/\
  --model_id FaceDetection \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --temperature 0.5 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 2 \
  --GE_num 2 \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 400 \
  --patience 30