import os

import torch
from tqdm import tqdm

from src.logger import Logger
from src.dataset import Titanic
from src.backbone import BasicBlock, Tree


class Trainer:
    def __init__(self, loss_fn, optimizer, epoch):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epoch = epoch

    def run_epoch(self, model, data_loader, phase='train', cb=None):
        num_iters = len(data_loader)

        avg_loss = 0

        pbar = tqdm(total=num_iters)

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                print('STOP')
                break
            full_iter = num_iters * (self.epoch - 1) + iter_id
            imgs, labels = tuple(batch)

            predict = model(imgs)
            loss = self.loss_fn(predict, labels)
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.detach().numpy()
                if callable(cb) and full_iter % 10 == 0:
                    cb({'loss': avg_loss / (iter_id + 1)}, full_iter, phase=phase)
            else:
                avg_loss += loss.detach().numpy()

            pbar.update()

        self.epoch += 1 if phase == 'train' else 0
        return avg_loss / num_iters

class params:
    lr = 0.005
    batch_size = 10
    num_workers = 4
    best_loss = 1e10
    num_epochs = 10
    start_epoch = 1
    val_interval = 1
    log_dir = './tmp/log'
    debug_dir = './tmp/debug'
    weight_dir = './tmp/weight'


def main():
    logger = Logger(params.log_dir, params.debug_dir, params=params)
    os.makedirs(params.weight_dir, exist_ok=True)
    model = BasicBlock(3, 5)
    model.eval()

    tmp_name = './tmp.onnx'
    im_tensor = torch.randn(1, 3, 768, 1024)
    res = model(im_tensor)
    input_names = ["input"]
    # torch.onnx.export(self, im_tensor, Path(tmp_name).resolve(), verbose=True,
    #                   input_names=input_names)

    # optimizer = torch.optim.Adam(model.parameters(), params.lr)
    # loss = torch.nn.BCELoss()
    #
    # val_loader = torch.utils.data.DataLoader(
    #     MafaFace(os.path.expanduser('~/data/mask/MAFA/test_faces')),
    #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    # )
    #
    # train_loader = torch.utils.data.DataLoader(
    #     MafaFace(os.path.expanduser('~/data/mask/MAFA/train_faces')),
    #     batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True
    # )
    #
    # trainer = Trainer(loss, optimizer, params.start_epoch)
    #
    # def cb(args, iteration, phase=''):
    #     for key, val in args.items():
    #         logger.scalar_summary(f'{phase}_{key}', val, iteration)
    #
    # for epoch in range(params.start_epoch, params.num_epochs + 1):
    #     model.train()
    #     tran_loss = trainer.run_epoch(model, train_loader, phase='train', cb=cb)
    #
    #     logger.write(f'epoch: {epoch} | tr_loss: {tran_loss:8f} | ')
    #
    #
    #     if params.val_interval > 0 and epoch % params.val_interval == 0:
    #         model.eval()
    #         val_loss = trainer.run_epoch(model, val_loader, phase='val', cb=cb)
    #         logger.scalar_summary('val_loss', val_loss, epoch)
    #         if val_loss < params.best_loss:
    #             params.best_loss = val_loss
    #             torch.save(model, os.path.join(params.weight_dir, 'model_best.pth'))
    #
    #             logger.write(f'val_loss: {val_loss:8f} | ')
    #     else:
    #         torch.save(model, os.path.join(params.weight_dir, f'model_{epoch}.pth'))
    #
    #     logger.write('\n')
    #
    # logger.close()


