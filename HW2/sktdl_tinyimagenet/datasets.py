import os
from urllib.request import urlretrieve
import torch
import torchvision


from sacred import Ingredient

tinyimagenet_ingredient = Ingredient('tinyimagenet')

@tinyimagenet_ingredient.config
def tinyimagenet_config():
    dataset_name = 'tiny-imagenet-200'
    download_path = '.'
    download_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

@tinyimagenet_ingredient.capture
def get_imagefolder(
        subset,
        download_path,
        dataset_name):
    assert subset in ['train', 'test', 'val']
    dataset_path = os.path.join(download_path, dataset_name, subset)
    if not os.path.exists(dataset_path):
        print('Downloading tinyimagenet')
        # download is only going to be dependency-injected
        # in experiment.py, thus binding manually
        download()
    assert os.path.exists(dataset_path)
    transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomRotation(20),
            # torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            ])
    image_folder = torchvision.datasets.ImageFolder(dataset_path, transforms)
    return image_folder


@tinyimagenet_ingredient.capture
def download(
        download_path,
        download_url,
        dataset_name):
    path = download_path
    url = download_url

    if os.path.exists(os.path.join(path, dataset_name, "val", "n01443537")):
        print("%s already exists, not downloading" % os.path.join(path, dataset_name))
        return
    else:
        print("Dataset not exists or is broken, downloading it")
    urlretrieve(url, os.path.join(path, dataset_name + ".zip"))
    
    import zipfile
    with zipfile.ZipFile(os.path.join(path, dataset_name + ".zip"), 'r') as archive:
        archive.extractall()

    # move validation images to subfolders by class
    val_root = os.path.join(path, dataset_name, "val")
    with open(os.path.join(val_root, "val_annotations.txt"), 'r') as f:
        for image_filename, class_name, _, _, _, _ in map(str.split, f):
            class_path = os.path.join(val_root, class_name)
            os.makedirs(class_path, exist_ok=True)
            os.rename(
                os.path.join(val_root, "images", image_filename),
                os.path.join(class_path, image_filename))

    os.rmdir(os.path.join(val_root, "images"))
    os.remove(os.path.join(val_root, "val_annotations.txt"))
