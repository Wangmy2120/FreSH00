export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/NATOPS/\
  --model_id NATOPS \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --temperature 2 \
  --Seg_num 3 \
  --SegE_num 3  \
  --GE_num 3 \
  --itr 5 \
  --learning_rate 0.01 \
  --train_epochs 100 \
  --patience 50