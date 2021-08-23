#DND train
CUDA_VISIBLE_DEVICES=-1 python main.py --dataset_dir /data3/sap/DND_NOISY --savename dnd 

#imagenet train
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode train --noise synth --savename imagenet

#set14 val
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode val_models --val_dataset set14 --savename imagenet

#set14 demo
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode demo --val_dataset set14 --savename imagenet_noise_clip_epoch005_result --model_path /data3/sap/nb2nb/imagenet_noise_clip/checkpoint/epoch005.pth
