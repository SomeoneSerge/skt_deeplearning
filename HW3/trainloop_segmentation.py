import torch

from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric
from ignite.handlers import ModelCheckpoint

from .iou import calc_iou


class IoU(Metric):
    def update(self, output):
        cpu = torch.device('cpu')
        y_pred, y = output['y_pred'].clone(), output['y'].clone()
        y_pred, y = y_pred.to(cpu).numpy(), y.to(cpu).numpy()
        self._iou = calc_iou(ground_truth=y_pred, prediction=y)

    def compute(self):
        return self._iou


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
                iou=IoU()
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
