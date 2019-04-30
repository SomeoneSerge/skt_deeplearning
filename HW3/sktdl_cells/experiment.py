import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
import tensorboardX
import pprint
import collections

from sktdl_cells import trainable_params as param_selection
from sktdl_cells.data_cells import CellsSegmentation, CellsTransform
from sktdl_cells.trainloop_segmentation import train
# from sktdl_cells.losses import dice_loss as neg_dice_coeff
# from pytorch_unet import dice_loss
# from sktdl_cells.diceloss_rogertrullo import dice_loss
from sktdl_cells.diceloss_issamlaradji import dice_loss
from pytorch_unet.unet.unet_model import UNet
import os


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(MODULE_DIR, 'runs')


ex = sacred.Experiment('sktdl_cells')
ex.observers.append(FileStorageObserver.create(RUNS_DIR))


@ex.capture
def get_trainable_named_params(net, trainable_params):
    METHODS = dict(
            fixed=param_selection.fixed,
            headtail=param_selection.headtail
            )
    paramsets = (METHODS[tp[0]](*tp[1:])(net) for tp in trainable_params)
    params = ((n, p) for pset in paramsets for n, p in pset)
    params = collections.OrderedDict(params)
    return params.items()

def get_trainable_params(net):
    return (p for n, p in get_trainable_named_params(net))

@ex.capture
def make_model(weights_path, device, trainable_params, random_init):
    net = UNet(3, 1)
    state = torch.load(weights_path, map_location='cpu')
    net.load_state_dict(state)
    for p in net.parameters():
        p.requires_grad_(False)
    for p in get_trainable_params(net):
        if random_init:
            # TODO: different init for translations and rotations
            stddev = np.prod(p.shape)
            stddev = np.sqrt(stddev)
            p.data.normal_(std=1./stddev)
        p.requires_grad_(True)
    net.to(torch.device(device))
    return net

@ex.capture
def make_data(subset, batch_size, clone_times, train_transform):
    train = subset == 'train'
    transform = CellsTransform(**train_transform) if train else None
    cells = CellsSegmentation(
            subset,
            clone_times=clone_times if train else 1,
            xy_transform=transform)
    return DataLoader(
            cells,
            batch_size=batch_size)

@ex.capture
def make_optimizer(model, adam_params):
    params = get_trainable_params(model)
    optimizer = Adam(params, **adam_params)
    return optimizer

@ex.config
def cfg0():
    clone_times=500
    weights_path = os.path.join(MODULE_DIR, 'pytorch_unet.pth')
    batch_size=50
    is_deconv = False
    num_input_channels = 11
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainable_params = [('headtail', 2, 2)]
    adam_params = dict(
            lr=1e-3,
            betas=(.9, .99))
    num_epochs = 5
    train_transform = dict(
            degrees=180.,
            translate=(1., 1.),
            scale=(.9, 1.1),
            crop_size=(64, 64))
    random_init=True
    epochs_per_checkpoint=2
    assume_negated_dice=False

@ex.command(unobserved=True)
def print_parameternames():
    net = make_model()
    output = (
            '\n'.join(
                '{:40}: {}'.format(
                    parname,
                    'trainable' if p.requires_grad else 'frozen')
                for parname, p in net.named_parameters()))
    print(output)


@ex.automain
def main(device, num_epochs, epochs_per_checkpoint, assume_negated_dice, _run):
    model = make_model()
    dataloader_train = make_data('train')
    dataloader_val = make_data('val')
    optimizer = make_optimizer(model)
    # loss = lambda yhat, y: neg_dice_coeff(y, yhat)
    if assume_negated_dice:
        loss = lambda yhat, y: 1.-dice_loss(yhat, y.float())
    else:
        loss = lambda yhat, y: dice_loss(yhat, y.float())
    device = torch.device(device)
    EXPERIMENT_DIR = os.path.join(RUNS_DIR, str(_run._id))
    tensorboard = tensorboardX.SummaryWriter(EXPERIMENT_DIR)
    def log_iou(iou, epoch):
        tensorboard.add_scalar('val.iou', iou, epoch)
    def log_trainloss(trainloss, iteration):
        tensorboard.add_scalar('train.loss', trainloss, iteration)
    train(model,
            dataloader_train,
            dataloader_val,
            optimizer,
            loss,
            device,
            num_epochs,
            log_trainloss=log_trainloss,
            log_iou=log_iou,
            weights_dir=EXPERIMENT_DIR,
            epochs_per_checkpoint=epochs_per_checkpoint)
