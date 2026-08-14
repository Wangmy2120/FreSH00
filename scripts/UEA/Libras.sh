export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/Libras/\
  --model_id Libras \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --temperature 3 \
  --Seg_num 3 \
  --SegE_num 3  \
  --GE_num 2 \
  --itr 5 \
  --learning_rate 0.01 \
  --train_epochs 400 \
  --patience 50