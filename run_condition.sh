python -m torch.distributed.launch --nproc_per_node 1 --master_port 12233 --use_env DDP_main_conditional.py \
  --lr 5e-5 \
  --batch_size 64 \
  --task_name 'QT' \
  --eval_steps 50 \
  --num_steps 100 \
  --batch_size 64 \
  --epochs 100 \
