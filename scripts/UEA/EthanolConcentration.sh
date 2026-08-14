export CUDA_VISIBLE_DEVICES=0


 python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/\
  --model_id EthanolConcentration \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 8 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 2 \
  --GE_num 2 \
  --d_model 16 \
  --temperature 0.5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.004 \
  --train_epochs 50 \
  --patience 30