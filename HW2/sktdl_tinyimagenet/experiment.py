import torch
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, TensorboardObserver
from sktdl_tinyimagenet import model, datasets
import math


# It's not a very respectable decision
# to fuse DI framework into the very heart of application
# but it allows faster bootstrap.
# Ideally, one should rather do something similar to
# `https://github.com/yuvalatzmon/SACRED_HyperOpt_v2/blob/master/sacred_wrapper.py`


ex = Experiment('sktdl_tinyimagenet')
ex.observers.append(FileStorageObserver.create('f_runs'))
ex.observers.append(TensorboardObserver('runs')) # make .creat() perhaps?

@ex.config
def config0():
    n_classes = 200
    layers_per_stage = 2
    widen_factor = 4
    drop_rate = 0.2
    apooling_cls = torch.nn.AdaptiveMaxPool2d
    apooling_output_size = (10, 10)

    n_epochs = 5
    optimizer_cls = torch.optim.Adam
    optimizer_params = dict(
            lr=.01,
            betas=[.9, .999]
            )
    loss_cls = torch.nn.CrossEntropyLoss


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    num_workers = 1
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
    return correct/total

get_loss = ex.capture(lambda loss_cls: loss_cls())

@ex.capture
def train(n_epochs, _run, device):
    print('Using device {device}'.format(device=device))
    dataset = get_dataset('train')
    net = get_network()
    net(next(iter(dataset))[0]) # For AdaptiveLinear to make weights
    net = net.to(device)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:

                a = 1.
                for s in p.shape:
                    a *= s
                a = 1./math.sqrt(a)
                p.uniform_(-a, a)
    optimizer = get_optimizer(net.parameters())
    loss = get_loss().to(device)
    print('Architecture:')
    print(net)
    print('Entering train loop!')
    it = 0
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
            with torch.no_grad():
                batch_acc = (yhat.argmax(-1) == y).sum().item()
                batch_acc = float(batch_acc)/float(X.shape[0])
            total_loss += obj.item()/float(X.shape[0])
            _run.log_scalar('batch.loss', obj.item(), it)
            _run.log_scalar('batch.loss', batch_acc, it)
            for name, p in net.named_parameters():
                _run.log_scalar('{}.grad'.format(name), torch.norm(p), it)
            it = it + 1
        _run.log_scalar('train.epoch.loss', total_loss, it) # aligning smoothened per-epoch plot and noisy per-iter plots
        # print('train.loss: {:.6f}'.format(total_loss))
        test_acc = evaluate(net, 'test')
        _run.log_scalar('{}.accuracy'.format(subset), test_acc, it)
        # TODO: make a metric-printing observer
        # print('{}.accuracy: {:.6f}'.format(subset, correct/total))
        # evaluate(net, 'train')
        #
        #
    filename = 'tmp_weights.pt'
    torch.save(
            net.state_dict(),
            filename
            )
    _run.add_artifact(filename, name='weights')
