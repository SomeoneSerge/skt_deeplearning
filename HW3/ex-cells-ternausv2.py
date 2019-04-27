import torch
import sacred
from torch.utils.data import DataLoader

from TernausNetV2.models.ternausnet2 import TernausNetV2
from data_cells import CellsSegmentation
from trainloop_segmentation import train
from losses import dice_loss as neg_dice_coeff


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


ex = sacred.Experiment('SegmentationViaTernausV2')

TernausNetV2 = ex.capture(TernausNetV2)

@ex.capture
def make_model(weights_path, device, trainable_params):
    net = TernausNetV2()
    state = torch.load(weights_path)
    net.load_state_dict(state)
    for name, p in net.named_parameters():
        p.requires_grad_(name in trainable_params)
    net.to(torch.device(device))

@ex.capture
def make_data(subset, batch_size, ):
    cells = CellsSegmentation()
    return DataLoader(cells, batch_size=batch_size)

@ex.capture
def make_optimizer(model, trainable_params, adam_params):
    params = tuple([p for n, p in model.named_parameters() if n in trainable_params])
    optimizer = Adam(params, **adam_params)
    return optimizer

@ex.config
def cfg0():
    weights_path = os.path.join(MODULE_DIR, 'ternaus_v2.pt')
    num_classes = 2
    num_filters = 32
    is_deconv = False
    num_input_channels = 11
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainable_params = tuple(['final']) # TODO: check if these params exist
    adam_params = dict(
            lr=1e-3,
            betas=(.9, .99)
            )
    num_epochs = 5

@ex.automain
def main(device, num_epochs):
    model = make_model()
    dataloader = make_data()
    optimizer = make_optimizer(model)
    loss = neg_dice_coeff
    device = torch.device(device)
    train(model, dataloader, optimizer, loss, device, num_epochs)
