source ~/.bashrc 
#export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29531 train_WAP_gpus.py