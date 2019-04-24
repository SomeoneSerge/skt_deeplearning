import torch
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, TensorboardObserver
from sktdl_tinyimagenet import model as my_model, datasets, xternalz_wideresnet
import math
import tqdm
import pprint


from .datasets import get_tinyimagenet, get_cifar10, get_cifar100, tinyimagenet_ingredient, cifar_ingredient

# It's not a very respectable decision
# to fuse DI framework into the very heart of application
# but it allows faster bootstrap.
# Ideally, one should rather do something similar to
# `https://github.com/yuvalatzmon/SACRED_HyperOpt_v2/blob/master/sacred_wrapper.py`


ex = Experiment('sktdl_tinyimagenet', ingredients=[tinyimagenet_ingredient, cifar_ingredient])
ex.observers.append(FileStorageObserver.create('f_runs'))
ex.observers.append(TensorboardObserver('runs')) # make .creat() perhaps?


make_wideresnet = ex.capture(my_model.make_wideresnet)
make_xternalz = ex.capture(
            lambda n_classes, depth, widen_factor, drop_rate, apooling_cls:
            xternalz_wideresnet.WideResNet(
                depth=depth,
                apooling_cls=apooling_cls,
                num_classes=n_classes,
                widen_factor=widen_factor,
                dropRate=drop_rate))

@ex.capture
def get_network(network_builder, append_logsoftmax):
    net = network_builder()
    if append_logsoftmax:
        net = torch.nn.Sequential(net, torch.nn.LogSoftmax(-1))
    return net

@ex.config
def config0():
    n_classes = 200
    batch_size = 100
    append_logsoftmax = False
    dataset = get_tinyimagenet
    n_epochs = 10
    depth = 16
    widen_factor = 4
    drop_rate = 0.02
    apooling_cls = torch.nn.AdaptiveMaxPool2d
    optimizer_cls = torch.optim.Adam
    adam_params = dict(
            lr=.001,
            betas=[.6, .999]
            )
    optimizer_params = ex.capture(lambda adam_params: adam_params)
    loss_cls = torch.nn.CrossEntropyLoss
    log_norms = False
    log_gradnorms = False
    num_workers = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_architecture = False

@ex.named_config
def wideresnet():
    network_builder = make_wideresnet
    resblock_strides = (1,2,3)
    make_conv = my_model.conv_bn_relu

@ex.named_config
def xternalz():
    network_builder = make_xternalz


@ex.named_config
def use_sgd():
    optimizer_cls = torch.optim.SGD
    sgd_params = dict(
            lr=.003,
            momentum=.9,
            nesterov=True
            )
    optimizer_params = ex.capture(lambda sgd_params: sgd_params)

@ex.named_config
def cifar10():
    dataset = get_cifar10

@ex.named_config
def cifar100():
    dataset = get_cifar100

@ex.capture
def get_optimizer(params, optimizer_cls, optimizer_params):
    return optimizer_cls(params, **optimizer_params())

@ex.capture
def get_dataloader(
        subset,
        batch_size,
        dataset,
        num_workers,):
    image_folder = dataset(subset=subset)
    batches = torch.utils.data.DataLoader(image_folder, batch_size, num_workers)
    return batches

@ex.capture
def evaluate(model, subset, device, _run):
    dataset = get_dataloader(subset)
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataset:
            y = y.to(device, non_blocking=True)
            X = X.to(device)
            correct += (model(X).argmax(-1) == y).sum().item()
            total += int(X.shape[0])
    return correct/total

get_loss = ex.capture(lambda loss_cls: loss_cls())

@ex.command
def print_shapes():
    dataset = get_dataloader('train')
    net = get_network()
    X = next(iter(dataset))[0]
    yhat = net(X)
    print('Input shapes are {}'.format(tuple(X.shape)))
    print('Output shapes are {}'.format(tuple(yhat.shape)))

@ex.capture
def train(
        n_epochs,
        device,
        log_norms,
        log_gradnorms,
        _run,
        optimizer_params,
        print_architecture):
    print('Using device {device}'.format(device=device))
    dataset = get_dataloader('train')
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
    net.train()
    if print_architecture:
        print('Architecture:')
        print(net)
    print('Number of parameters: {}'.format(sum(p.numel() for p in net.parameters())))
    print('Entering train loop')
    it = 0
    for e in range(n_epochs):
        total_loss = 0.
        with tqdm.tqdm(
                enumerate(dataset),
                desc='[{}/{}]'.format(e, n_epochs)) as pbar:
            for b, (X, y) in enumerate(dataset):
                y = y.to(device, non_blocking=True)
                X = X.to(device)
                optimizer.zero_grad()
                yhat = net(X)
                obj = loss(yhat, y)
                obj.backward()
                optimizer.step()
                try:
                    net.eval()
                    with torch.no_grad():
                        batch_acc = (yhat.argmax(-1) == y).sum().item()
                        batch_acc = float(batch_acc)/float(X.shape[0])
                    total_loss += obj.item()/float(X.shape[0])
                    _run.log_scalar('batch.loss', obj.item(), it)
                    _run.log_scalar('batch.accuracy', batch_acc, it)
                    if not (log_norms or log_gradnorms):
                        continue
                    with torch.no_grad():
                        for name, p in net.named_parameters():
                            if log_gradnorms:
                                _run.log_scalar('gradnorm__{}'.format(name), torch.norm(p.grad.data), it)
                            if log_norms:
                                _run.log_scalar('norm__{}'.format(name), torch.norm(p.data), it)
                finally:
                    net.train()
                    it = it + 1
                    pbar.set_postfix_str('batch.accuracy: {:.5f}'.format(batch_acc))
                    pbar.update(1)
        _run.log_scalar('train.loss', total_loss, it) # aligning smoothened per-epoch plot and noisy per-iter plots
        # print('train.loss: {:.6f}'.format(total_loss))
        test_acc = evaluate(net, 'test')
        _run.log_scalar('test.accuracy', test_acc, it)
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
