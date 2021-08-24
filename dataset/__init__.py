from option import args
from dataset.dataset import Dataset
from dataset.imagenet import Imagenet
from dataset.sidd_dataset import SIDD_dataset

def get_dataset(args, name):
    dataset = None
    if name == 'imagenet':
        dataset = Imagenet(args.imagenet_path)
    elif name == 'kodak':
        dataset = Dataset(args.kodak_dir)
    elif name == 'set14':
        dataset = Dataset(args.set14_dir)
    elif name == 'dnd':
        dataset = Dataset(args.dnd_dir)
    elif name == 'sidd':
        dataset = SIDD_dataset(args.noisy_dir, args.clean_dir)
    elif name == 'sidd_val':
        dataset = SIDD_dataset(args.val_noisy_dir, args.val_clean_dir)
    return dataset
