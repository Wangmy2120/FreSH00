export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandWriting/\
  --model_id Handwriting \
  --model FreSH \
  --data UEA \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --temperature 0.5 \
  --Seg_num 3 \
  --SegE_num 2  \
  --GE_num 2 \
  --itr 20 \
  --learning_rate 0.0005 \
  --train_epochs 400 \
  --patience 50