import torch
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sktdl_tinyimagenet import model, datasets


# It's not a very respectable decision
# to fuse DI framework into the very heart of application
# but it allows faster bootstrap.
# Ideally, one should rather do something similar to
# `https://github.com/yuvalatzmon/SACRED_HyperOpt_v2/blob/master/sacred_wrapper.py`


ex = Experiment('sktdl_tinyimagenet')
ex.observers.append(FileStorageObserver.create('runs'))

@ex.config
def config0():
    n_classes = 200
    layers_per_stage = 1
    widen_factor = 3
    drop_rate = 0.2

    n_epochs = 2
    optimizer_cls = torch.optim.Adam
    optimizer_params = dict(
            lr=1e-3,
            betas=[.9, .999]
            )
    loss_cls = torch.nn.CrossEntropyLoss


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_workers = 2
    dataset_name = 'tiny-imagenet-200'
    download_path = '.'
    download_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

get_network = ex.capture(model.make_wideresnet)
get_dataset = ex.capture(datasets.get_dataset)

@ex.capture
def get_optimizer(params, optimizer_cls, optimizer_params):
    return optimizer_cls(params, **optimizer_params)

@ex.capture
def evaluate(model, subset, device, _run):
    dataset = get_dataset(subset)
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataset:
            y = y.to(device, non_blocking=True)
            X = X.to(device)
            correct += (model(X).argmax(-1) == y).sum().item()
            total += int(X.shape[0])
    _run.log_scalar('{}.accuracy'.format(subset), correct/total)
    # TODO: make a metric-printing observer
    print('{}.accuracy: {.6f}'.format(subset, correct/total))

get_loss = ex.capture(lambda loss_cls: loss_cls())

@ex.capture
def train(n_epochs, _run, device):
    dataset = get_dataset('train')
    net = get_network()
    net(next(iter(dataset))[0]) # For AdaptiveLinear to make weights
    net = net.to(device)
    optimizer = get_optimizer(net.parameters())
    loss = get_loss().to(device)
    print('INITed!')
    for e in range(n_epochs):
        total_loss = 0.
        for b, (X, y) in enumerate(dataset):
            y = y.to(device, non_blocking=True)
            X = X.to(device)
            optimizer.zero_grad()
            yhat = net.forward(X)
            obj = loss(yhat, y)
            obj.backward()
            optimizer.step()
            total_loss += obj.item()/float(X.shape[0])
        _run.log_scalar('train.loss', total_loss)
        evaluate(net, 'test')
        evaluate(net, 'train')
    filename = 'tmp_weights.pt'
    torch.save(
            net.state_dict(),
            filename
            )
    _run.add_artifact(filename, name='weights')
