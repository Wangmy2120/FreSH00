export CUDA_VISIBLE_DEVICES=0


   python -u runFreSH.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/\
  --model_id JapaneseVowels \
  --model FreSH \
  --data UEA \
  --batch_size 32 \
  --d_model 32 \
  --itr 1 \
  --MoE_flag 4 \
  --Seg_num 3 \
  --SegE_num 2 \
  --GE_num 2 \
  --temperature 1 \
  --itr 20 \
  --learning_rate 0.005 \
  --train_epochs 400 \
  --patience 50