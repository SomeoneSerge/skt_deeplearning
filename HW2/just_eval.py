import torch
import numpy as np
from argparse import ArgumentParser
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import sys

from sktdl_tinyimagenet.model import make_wideresnet, conv_bn_relu

def cpu():
    return torch.device('cpu')

def device():
    return torch.device('cuda') if torch.cuda.is_available() else cpu()

def make_model():
    model = make_wideresnet(
            n_classes=200,
            depth=16,
            make_conv=conv_bn_relu,
            apooling_cls=torch.nn.modules.pooling.AdaptiveMaxPool2d,
            resblock_strides=(1, 2, 3),
            widen_factor=4,
            drop_rate=.2)
    model = torch.nn.Sequential(model, torch.nn.LogSoftmax(-1))
    return model

def evaluate(model, data):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in data:
            y = y.to(device(), non_blocking=True)
            X = X.to(device())
            correct += float((model(X).argmax(-1) == y).sum().item())
            total += int(X.shape[0])
    return correct/total

def make_dataloader(image_folder, batch_size):
    image_folder = ImageFolder(image_folder, ToTensor())
    batches = torch.utils.data.DataLoader(image_folder, batch_size)
    return batches

def cmd_evaluate(args):
    model = make_model()
    model.load_state_dict(torch.load(args.weights, map_location=device()))
    data = make_dataloader(args.images_path, args.batch_size)
    score = evaluate(model, data)
    print(score)

def cmd_print_architecture(args):
    model = make_model()
    print(model)


if __name__ == '__main__':
    parser = ArgumentParser('just_eval')
    subparsers = parser.add_subparsers()
    p_eval = subparsers.add_parser('evaluate')
    p_eval.add_argument(
            '--weights',
            help='Path to trained weights',
            default='runs/15/weights')
    p_eval.add_argument(
            '--batch_size',
            type=int,
            default=1)
    p_eval.add_argument(
            '--images-path',
            default='tiny-imagenet-200/val')
    p_eval.set_defaults(cmd=cmd_evaluate)
    args = parser.parse_args()
    p_archi = subparsers.add_parser('print-architecture')
    p_archi.set_defaults(cmd=cmd_print_architecture)

    args = parser.parse_args()
    if not hasattr(args, 'cmd'):
        parser.print_help()
        sys.exit(1)
    args.cmd(args)
