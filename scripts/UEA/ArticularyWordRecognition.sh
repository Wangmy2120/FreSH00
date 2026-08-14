export CUDA_VISIBLE_DEVICES=0



  python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UEA1/UEA/ArticularyWordRecognition/\
  --model_id ArticularyWordRecognition \
  --model FreSH \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --MoE_flag 4 \
  --temperature 0.5 \
  --Seg_num 3 \
  --SegE_num 1  \
  --GE_num 1 \
  --itr 1 \
  --learning_rate 0.005 \
  --train_epochs 100 \
  --patience 50
