import torch

import sys
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric
from ignite.metrics import MetricsLambda
from ignite.handlers import ModelCheckpoint

import pprint


# In the end, ignite's trainloop looks very much like shit,
# heavily coupling the trainloop itself and IO bullshit
# like logging, in the meantime making you bind everything
# to its interfaces.



class IoU(Metric):
    def __init__(self, impl):
        super(IoU, self).__init__()
        self.iou_impl = impl
    def update(self, output):
        cpu = torch.device('cpu')
        y_pred, y = output
        self._iou = self.iou_impl(y_pred, y)
    def compute(self):
        return self._iou
    def reset(self):
        self._iou = 0.


def train(
        model,
        trainloader,
        valloader,
        optimizer,
        loss,
        iou,
        device,
        num_epochs,
        log,
        weights_dir,
        epochs_per_checkpoint):
    def update(engine, batch):
        print('update()')
        optimizer.zero_grad()
        x, y = batch
        yhat = model(x)
        J = loss(yhat, y)
        J.backward()
        optimizer.step()
        return dict(
                loss=J,
                y_pred=yhat,
                y=y
                )
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(
            model,
            metrics=dict(
                iou=IoU(iou)
                ),
            device=device)
    @trainer.on(Events.EPOCH_COMPLETED)
    def _on_epoch(trainer):
        evaluator.run(valloader)
        log('val', 'iou', evaluator.state.metrics['iou'], trainer.state.epoch)
    @trainer.on(Events.ITERATION_COMPLETED)
    def _on_iter(trainer):
        log('train', 'loss', trainer.state.output, trainer.state.iteration)
    checkpointer = ModelCheckpoint(
            weights_dir,
            'weights',
            save_interval=epochs_per_checkpoint,
            n_saved=2,
            create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, dict(model=model))
    trainer.run(trainloader, max_epochs=num_epochs)
