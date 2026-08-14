export CUDA_VISIBLE_DEVICES=0


  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/PhonemeSpectra/\
  --model_id PhonemeSpectra \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 16 \
  --MoE_flag 1 \
  --temperature 1 \
  --Seg_num 3 \
  --SegE_num 2  \
  --GE_num 2 \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 30