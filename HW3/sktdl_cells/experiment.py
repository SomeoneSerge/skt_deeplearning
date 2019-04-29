import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import sacred
from sacred.observers import FileStorageObserver
import tensorboardX
import pprint

from sktdl_cells.data_cells import CellsSegmentation
from sktdl_cells.trainloop_segmentation import train
# from sktdl_cells.losses import dice_loss as neg_dice_coeff
from pytorch_unet import dice_loss
from pytorch_unet.unet.unet_model import UNet
import os


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


ex = sacred.Experiment('sktdl_cells')
ex.observers.append(FileStorageObserver.create(os.path.join(MODULE_DIR, 'runs')))


@ex.capture
def make_model(weights_path, device, trainable_params):
    net = UNet(3, 1)
    state = torch.load(weights_path, map_location='cpu')
    net.load_state_dict(state)
    for name, p in net.named_parameters():
        p.requires_grad_(name in trainable_params)
    net.to(torch.device(device))
    return net

@ex.capture
def make_data(subset, batch_size, ):
    cells = CellsSegmentation(subset)
    return DataLoader(cells, batch_size=batch_size)

@ex.capture
def make_optimizer(model, trainable_params, adam_params):
    params = tuple([p for n, p in model.named_parameters() if n in trainable_params])
    optimizer = Adam(params, **adam_params)
    return optimizer

@ex.config
def cfg0():
    weights_path = os.path.join(MODULE_DIR, 'pytorch_unet.pth')
    batch_size=5
    is_deconv = False
    num_input_channels = 11
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainable_params = tuple([
        'outc.conv.weight',
        'outc.conv.bias'
        ]) # TODO: check if these params exist
    adam_params = dict(
            lr=1e-3,
            betas=(.9, .99)
            )
    num_epochs = 5

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
def main(device, num_epochs):
    model = make_model()
    dataloader = make_data('train')
    optimizer = make_optimizer(model)
    # loss = lambda yhat, y: neg_dice_coeff(y, yhat)
    loss = lambda yhat, y: 1. - dice_loss.dice_coeff(yhat, y.float())
    device = torch.device(device)
    tensorboard = tensorboardX.SummaryWriter('runs')
    def log_iou(iou, epoch):
        tensorboard.add_scalar('val.iou', iou, epoch)
    def log_trainloss(trainloss, iteration):
        tensorboard.add_scalar('train.loss', trainloss, iteration)
    train(model, dataloader, optimizer, loss, device, num_epochs, log_trainloss, log_iou)
