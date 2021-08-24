import argparse 
import template

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train')

parser.add_argument('--noise', default='real')

parser.add_argument('--model', default='unet')

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', default=4)
parser.add_argument('--shuffle', action="store_true", default=True)
parser.add_argument('--num_workers', default=0)
#parser.add_argument('--lambda_ratio', default=1)

parser.add_argument('--basepath', default='/data/hhh7748/deformable_nb2nb')
parser.add_argument('--savename')

parser.add_argument('--train_dataset', default='sidd')
parser.add_argument('--val_dataset', default='sidd_val')
parser.add_argument('--model_path', default=None)

parser.add_argument('--imagenet_path', default='/data3/sjyang/dataset/imagenet_val.h5')
parser.add_argument('--kodak_dir', default='/data3/sap/dataset/kodak')
parser.add_argument('--set14_dir', default='/data3/sap/dataset/Set14')
parser.add_argument('--dnd_dir', default='/data3/sap/dataset/DND_NOISY')
parser.add_argument('--noisy_dir', default='/data/hhh7748/SIDD_new_Cropped/noisy')
parser.add_argument('--clean_dir', default='/data/hhh7748/SIDD_new_Cropped/clean')
#parser.add_argument('--val_noisy_dir', default='/data/hhh7748/sidd_val/noisy')
#parser.add_argument('--val_clean_dir', default='/data/hhh7748/sidd_val/clean')
parser.add_argument('--val_noisy_dir', default='/data/hhh7748/SIDD_new_Cropped/noisy')
parser.add_argument('--val_clean_dir', default='/data/hhh7748/SIDD_new_Cropped/clean')



args = parser.parse_args()
template.set_template(args)
