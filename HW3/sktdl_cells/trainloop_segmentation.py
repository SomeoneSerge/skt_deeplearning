import torch

import sys
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric
from ignite.metrics import MetricsLambda
from ignite.handlers import ModelCheckpoint

from sktdl_cells.iou import calc_iou


# In the end, ignite's trainloop looks very much like shit,
# heavily coupling the trainloop itself and IO bullshit
# like logging, in the meantime making you bind everything
# to its interfaces.



def calc_iou_torch(y, y_pred):
    cpu = torch.device('cpu')
    y_pred, y = output['y_pred'].clone(), output['y'].clone()
    y_pred, y = y_pred.to(cpu).numpy(), y.to(cpu).numpy()
    self._iou = calc_iou(ground_truth=y, prediction=y_pred)

IoU = MetricsLambda(
        calc_iou_torch,
        lambda output: (output['y'], output['y_pred']))


def train(
        model,
        dataloader,
        optimizer,
        loss,
        device,
        num_epochs,
        log_trainloss,
        log_iou):
    def update(engine, batch):
        print('udpate()')
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
                iou=IoU
                ),
            device=device)
    @trainer.on(Events.EPOCH_COMPLETED)
    def _on_epoch(trainer):
        evaluator.run(validation_set)
        log_iou(trainer.state.metrics['iou'], trainer.state.epoch)
    @trainer.on(Events.ITERATION_COMPLETED)
    def _on_iter(trainer):
        log_trainloss(trainer.state.output, trainer.state.iteration)
    trainer.run(dataloader, max_epochs=num_epochs)
