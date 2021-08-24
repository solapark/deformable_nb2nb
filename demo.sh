#DND train
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode train --noise real --savename dnd --train_dataset dnd   

#DND val
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode val_models --savename dnd --val_dataset dnd 

#imagenet train
CUDA_VISIBLE_DEVICES=-1 python -m pdb main.py --mode train --noise synth --savename imagenet

#set14 val
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode val_models --val_dataset set14 --savename imagenet

#set14 demo
CUDA_VISIBLE_DEVICES=3 python -m pdb main.py --mode demo --val_dataset set14 --savename imagenet/ep065 --model_path /data3/sap/nb2nb/imagenet/checkpoint/epoch065.pth
