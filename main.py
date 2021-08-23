import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import tensorflow as tf 

import os
import cv2
import numpy as np
from tqdm import tqdm

from synthetic_data_generator import *
from image_sub_sampler import *
from models import MODELS
from dataset import get_dataset
from option import args 

def get_concat_img(img_list, cols=3):
    rows = int(len(img_list)/cols)
    hor_imgs = [np.hstack(img_list[i*cols:(i+1)*cols]) for i in range(rows)]
    ver_imgs = np.vstack(hor_imgs)
    return ver_imgs

def clip_to_uint8(arr):
    if isinstance(arr, np.ndarray):
        return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    x = tf.clip_by_value(arr * 255.0 + 0.5, 0, 255)
    return tf.cast(x, tf.uint8)

def calculate_psnr(a, b, axis=None):
    a, b = [clip_to_uint8(x) for x in [a, b]]
    if isinstance(a, np.ndarray):
        a, b = [x.astype(np.float32) for x in [a, b]]
        x = np.mean((a - b)**2, axis=axis)
        return np.log10((255 * 255) / x) * 10.0
    a, b = [tf.cast(x, tf.float32) for x in [a, b]]
    x = tf.reduce_mean((a - b)**2, axis=axis)
    return tf.log((255 * 255) / x) * (10.0 / math.log(10))

def train(args, device, network, noise_adder):
    log_dir = os.path.join(args.basepath, args.savename, 'log')
    os.makedirs(log_dir, exist_ok=True) 
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    dataset = get_dataset(args, args.train_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    #val_dataset = get_dataset(args, args.val_dataset)
    #val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    dir_checkpoint = os.path.join(args.basepath, args.savename, 'checkpoint')
    os.makedirs(dir_checkpoint, exist_ok=True) 

    #dir_result = os.path.join(args.basepath, args.savename, 'result')
    #os.makedirs(dir_result, exist_ok=True) 

    n_epoch = args.epoch
    for epoch in range(1, n_epoch+1):
        print('epoch', epoch, '/', n_epoch)
        for i, clean in enumerate(dataloader):
            if(i+1) % 50 == 0 :
                print( i+1, '/', len(dataloader))
            # preparing synthetic noisy images
            clean = clean / 255.0
            clean = clean.to(device = device)
            #noisy = noise_adder.add_train_noise(clean)
            noisy = noise_adder.add_valid_noise(clean)
            optimizer.zero_grad()

            # generating a sub-image pair
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)

            # preparing for the regularization term
            with torch.no_grad():
                noisy_denoised = network(noisy)

            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            # calculating the loss 
            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2

            '''
            if i % 1 == 0 :
                noisy_d = (noisy[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                clean_d = (clean[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                noisy_denoised_d = (noisy_denoised[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                img_list = [noisy_d, clean_d, noisy_denoised_d]
                img_list =  get_concat_img(img_list, cols=3)
                cv2.imshow('noisy, clean, denoised', img_list)

                noisy_sub1_d = (noisy_sub1[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                noisy_sub2_d = (noisy_sub2[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                noisy_output_d = (noisy_output[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
                img_list = [noisy_sub1_d, noisy_sub2_d, noisy_output_d]
                img_list =  get_concat_img(img_list, cols=3)
                cv2.imshow('noisy_sub1, noisy_sub2, noisy_su1_denoised', img_list)
                cv2.waitKey()
            '''

            Lambda = epoch / n_epoch * args.lambda_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised
            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = loss1 + loss2
            #loss_all = loss1
            #print('loss1', loss1.item(), 'loss2', loss2.item(), 'loss_all', loss_all.item())
            loss_all.backward()
            optimizer.step()

            writer.add_scalar("Loss/loss_rec", loss1, epoch)
            writer.add_scalar("Loss/loss_reg", loss2, epoch)
            writer.add_scalar("Loss/loss_all", loss_all, epoch)
            
        scheduler.step()

        #for param_group in optimizer.param_groups:
        #    print(param_group['lr'])

        torch.save(network.state_dict(), os.path.join(dir_checkpoint, 'epoch%03d.pth'%(epoch)))

        #cur_psnr = validation(network, noise_adder, val_dataloader, device)
        #print('ep', epoch, 'psnr', cur_psnr)
        #dir_result_ep = os.path.join(dir_result, 'ep%03d'%(epoch+1))
        #demo(network, noise_adder, val_dataloader, device, dir_result_ep)

def validation_all_epoch(args, network, noise_adder, device):
    dataset = get_dataset(args, args.val_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    dir_checkpoint = os.path.join(args.basepath, args.savename, 'checkpoint')
    dir_list = sorted(os.listdir(dir_checkpoint))
    all_models_path = [os.path.join(dir_checkpoint, filename) for filename in dir_list] 

    max_psnr = -1
    max_psnr_path = -1
    for model_path in all_models_path :
        network.load_state_dict(torch.load(model_path))
        cur_psnr = validation(network, noise_adder, dataloader, device)
        print('cur_psnr', cur_psnr, 'model_path', model_path)
        if cur_psnr > max_psnr :
            max_psnr = cur_psnr
            max_psnr_path = model_path
            
    print('max_psnr', max_psnr, 'max_psnr_path', max_psnr_path)

    dir_result = os.path.join(args.basepath, args.savename, 'result', args.val_dataset)
    demo(network, noise_adder, dataloader, device, dir_result)

def validation(network, noise_adder, dataloader, device):
    psnr_sum = 0
    for i, clean_raw in enumerate(dataloader):
        clean = clean_raw / 255.0
        clean = clean.to(device = device)
        noisy = noise_adder.add_valid_noise(clean)
        noisy_denoised = network(noisy)

        psnr =  calculate_psnr(clean.detach().cpu().numpy(), noisy_denoised.detach().cpu().numpy(), axis=(1,2,3))
        '''
        for c in range(3):
            psnr =  calculate_psnr(clean[:, c:c+1].detach().cpu().numpy(), noisy_denoised[:, c:c+1].detach().cpu().numpy(), axis=(1,2,3))
            print(i, psnr)
        print('')
        '''
        psnr_sum += psnr
    return  psnr_sum/len(dataloader)

def demo(network, noise_adder, dataloader, device, dir_result):
    os.makedirs(dir_result, exist_ok=True) 

    for i, clean in enumerate(dataloader):
        clean = clean / 255.0
        clean = clean.to(device = device)
        noisy = noise_adder.add_valid_noise(clean)
        noisy_denoised = network(noisy)
       
        
        path = os.path.join(dir_result, '%04d.jpg'%(i))

        clean_d = (clean[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
        noisy = (noisy[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
        noisy_denoised = (noisy_denoised[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)

        img_list = [noisy, clean_d, noisy_denoised]
        result_noisy_clean_denoised =  get_concat_img(img_list, cols=3)
        cv2.imwrite(path, result_noisy_clean_denoised)

        '''
        clean_path = os.path.join(dir_result, '%04d_clean.jpg'%(i))
        noisy_path = os.path.join(dir_result, '%04d_noisy.jpg'%(i))
        denoised_path = os.path.join(dir_result, '%04d_denoised.jpg'%(i))

        cv2.imwrite(clean_path, clean_raw[0].numpy().astype('uint8').transpose(1, 2, 0))
        cv2.imwrite(noisy_path, (noisy[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0))
        cv2.imwrite(denoised_path, (noisy_denoised[0]*255.0).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0))
        '''

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = MODELS[args.model](n_channels=3, n_classes=3)
    network.to(device=device)
    if args.model_path :
        network.load_state_dict(torch.load(args.model_path, map_location=device))

    noise_adder = AugmentNoise(style="gauss25")

    if args.mode == 'train':
        train(args, device, network, noise_adder)
    elif args.mode == 'val_models':
        validation_all_epoch(args, network, noise_adder, device)
    elif args.mode == 'demo':
        dataset = get_dataset(args, args.val_dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dir_result = os.path.join(args.basepath, args.savename, 'result', args.val_dataset)
        demo(network, noise_adder, dataloader, device, dir_result)
