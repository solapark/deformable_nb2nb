from option import args
from dataset.dataset import Dataset
from dataset.imagenet import Imagenet

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
    return dataset
