export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/LSST/\
  --model_id LSST \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 4 \
  --temperature 0.5 \
  --Seg_num 2 \
  --SegE_num 2  \
  --GE_num 2 \
  --itr 10 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 50