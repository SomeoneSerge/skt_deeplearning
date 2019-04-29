import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric
from ignite.metrics import MetricsLambda
from ignite.handlers import ModelCheckpoint

from sktdl_cells.iou import calc_iou



def calc_iou_torch(y, y_pred):
    cpu = torch.device('cpu')
    y_pred, y = output['y_pred'].clone(), output['y'].clone()
    y_pred, y = y_pred.to(cpu).numpy(), y.to(cpu).numpy()
    self._iou = calc_iou(ground_truth=y_pred, prediction=y)

IoU = MetricsLambda(
        calc_iou_torch,
        lambda output: (output['y'], output['y_pred']))


def train(model, dataloader, optimizer, loss, device, num_epochs):
    def update(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        yhat = model(x)
        J = loss(y, yhat)
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
    def print_iou(trainer):
        print('[{:3}/{:3}] iou={:.5f}'
                .format(
                    trainer.state.epoch,
                    num_epochs,
                    trainer.state.metrics['iou']))
    trainer.run(dataloader, max_epochs=num_epochs)