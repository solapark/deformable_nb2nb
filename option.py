import argparse 
import template

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train')

parser.add_argument('--noise', default='synth')

parser.add_argument('--model', default='unet')

parser.add_argument('--epoch', default=100)
parser.add_argument('--batch_size', default=4)
parser.add_argument('--shuffle', action="store_true", default=True)
parser.add_argument('--num_workers', default=0)
parser.add_argument('--lambda_ratio', default=1)

parser.add_argument('--basepath', default='/data3/sap/nb2nb')
parser.add_argument('--savename')

parser.add_argument('--train_dataset', default='imagenet')
parser.add_argument('--val_dataset', default='kodak')
parser.add_argument('--model_path', default=None)

parser.add_argument('--imagenet_path', default='/data3/sjyang/dataset/imagenet_val.h5')
parser.add_argument('--kodak_dir', default='/data3/sap/dataset/kodak')
parser.add_argument('--set14_dir', default='/data3/sap/dataset/Set14')
parser.add_argument('--dnd_dir', default='/data3/sap/dataset/DND_NOISY')

args = parser.parse_args()
template.set_template(args)
