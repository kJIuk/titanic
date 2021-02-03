import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import Titanic
from src.logger import Logger
from src.model import Model


class Trainer:
    def __init__(self, loss_fn, optimizer, epoch):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epoch = epoch

    def run_epoch(self, model, data_loader, phase='train', cb=None):
        num_iters = len(data_loader)
        res = {
            'loss': 0,
            'acc': 0,
        }
        tp = 0

        # pbar = tqdm(total=num_iters)

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                print('STOP')
                break
            full_iter = num_iters * (self.epoch - 1) + iter_id
            imgs, labels = tuple(batch)
            labels = labels.unsqueeze(1).float()

            predict = model(imgs)

            loss = self.loss_fn(predict, labels)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if callable(cb) and full_iter % 10 == 0:
                #     cb({'av_loss': avg_loss / (iter_id + 1)}, full_iter, phase=phase)
                cb({'loss': loss}, full_iter, phase=phase)
            elif phase == 'val':
                tp += int(int(predict > 0.5) == labels)


            res['loss'] += loss.detach().numpy()

            # pbar.update()

        self.epoch += 1 if phase == 'train' else 0
        res['acc'] = tp / num_iters
        res['loss'] /= num_iters

        return res


def prepare_data(csv_file, cross=False):
    table_data = pd.read_csv(os.path.expanduser(csv_file))

    r = None if cross else 111
    X_train, X_test = train_test_split(table_data, test_size=0.2, train_size=0.8, random_state=r, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        Titanic(X_test),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )
    print("Validation: ", len(val_loader))

    train_loader = torch.utils.data.DataLoader(
        Titanic(X_train),
        batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True
    )
    print("Training: ", len(train_loader))
    return train_loader, val_loader


class params:
    lr = 0.001
    lr_step = [100, 150, 175]
    batch_size = 8
    num_workers = 1
    best_acc = 0
    num_epochs = 200
    start_epoch = 1
    val_interval = 1
    weight_decay = 0.001
    log_dir = './tmp/log'
    debug_dir = './tmp/debug'
    weight_dir = './tmp/weight'


def main():

    logger = Logger(params.log_dir, params.debug_dir, params=params)
    os.makedirs(params.weight_dir, exist_ok=True)
    input_size = 8
    model = Model(input_size)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), params.lr, weight_decay=params.weight_decay, amsgrad=True, betas=(0.1, 0.1))
    # optimizer = torch.optim.SGD(model.parameters(), params.lr, weight_decay=params.weight_decay, momentum=0.1)
    loss = torch.nn.BCELoss()

    trainer = Trainer(loss, optimizer, params.start_epoch)

    def cb(args, iteration, phase=''):
        for key, val in args.items():
            logger.scalar_summary(f'{phase}_{key}', val, iteration)

    best_epoch = 0
    resume = True
    for epoch in tqdm(range(params.start_epoch, params.num_epochs + 1)):
        if resume:
            model = torch.load(os.path.join(params.weight_dir, 'model_best.pth'))
        model.train()

        train_loader, val_loader = prepare_data('~/data/kaggle/titanic/train.csv', cross=True)

        res = trainer.run_epoch(model, train_loader, phase='train', cb=cb)

        logger.write(f'epoch: {epoch} | tr_loss: {res["loss"]:8f} | ')
        print(f'\nepoch: {epoch} | tr_loss: {res["loss"]:8f}')

        if params.val_interval > 0 and epoch % params.val_interval == 0:
            model.eval()
            res = trainer.run_epoch(model, val_loader, phase='val', cb=cb)
            logger.scalar_summary('acc', res["acc"], epoch)
            if res["acc"] > params.best_acc:
                params.best_acc = res["acc"]
                best_epoch = epoch
                torch.save(model, os.path.join(params.weight_dir, 'model_best.pth'))

            logger.write(f'acc: {res["acc"]:8f} | ')
            print(f'val_acc: {res["acc"]:8f}')
        else:
            torch.save(model, os.path.join(params.weight_dir, f'model_{epoch}.pth'))

        if epoch in params.lr_step:
            torch.save(model, os.path.join(params.weight_dir, f'model_{str(epoch)}.pth'))
            lr = params.lr * (0.1 ** (params.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            resume = True

        logger.write('\n')

    print('best_epoch: ', best_epoch, '| best_acc', params.best_acc)
    logger.close()



