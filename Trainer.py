import torch
from tqdm import tqdm
from utils.misc import AverageMeter, ScalerMeter
import os

class StandardTrainer():
    def __init__(self,manager, model, dataloaders, iters_per_epoch,criterion, optimizer, scheduler,
                    num_epochs, device, log_period=5, test_period=1):
        self.manager = manager
        self.logger = manager.get_logger()
        self.msg = ''
        self.last_log_iter_id = -1
        self.iters_per_epoch = iters_per_epoch
        self.tqdms = [None for _ in iters_per_epoch]
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler =scheduler
        self.num_epochs = num_epochs
        self.data_iters = [iter(loader) for loader in dataloaders]
        self.data_counters = [0 for _ in dataloaders]
        self.test_period = test_period
        self.log_period = log_period
        self.ckpt_period = 10000  # Only save the model in the end

        self.meters = {}
        self.meters_info = {}
        self.iter_count = 0

        self.add_meter('learning_rate', 'lr', meter_type='scaler')
        self.add_meter('loss', 'L', fstr_format='6.3f')
        self.add_meter('acc_train_clean', 'TrCl', meter_type='avg', fstr_format='5.3f', num_classes=10)
        self.add_meter('acc_test_clean', 'TeCl', meter_type='avg', fstr_format='5.3f', num_classes=10)

    def add_meter(self, name, abbr=None, meter_type='avg', fstr_format=None, num_classes=None):
        assert meter_type in ('avg', 'scaler', 'per_class_avg')
        self.meters_info[name] = {
            'abbr': abbr if abbr is not None else name,
            'type': meter_type,
            'format': fstr_format,
            'num_classes': num_classes,
        }

    def update_meters(self, training):
        self.meters['learning_rate'].update(self.scheduler.get_last_lr()[0])


    def _update_acc_meter(self, meter_name, predictions, labels):
        self.meters[meter_name].update(
            predictions.eq(labels).sum(), labels.size(0))

    def _setup_tqdms(self, training):
        idx = 0 if training else 1
        t = tqdm(total=self.iters_per_epoch[idx], leave=True, dynamic_ncols=True)
        t.clear()
        self.tqdms[idx] = t

    def _next_data_batch(self, idx):
        try:
            batch = next(self.data_iters[idx])
        except StopIteration:
            # Reset
            self.data_counters[idx] += 1
            loader = self.dataloaders[idx]
            self.data_iters[idx] = iter(loader)
            batch = next(self.data_iters[idx])
        return batch

    def get_checkpoint(self, epoch_id):
        model = self.model
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch_id,
        }
        return checkpoint

    def save_checkpoint(self, epoch_id, checkpoint_name=None):
        checkpoint = self.get_checkpoint(epoch_id)
        if checkpoint_name is None:
            name = f'ckpt-{epoch_id}.pt'
        else:
            name = f'{checkpoint_name}.pt'
        model_path = os.path.join(self.manager.get_checkpoint_dir(), name)
        torch.save(checkpoint, model_path)
        self.logger.info(f'Model saved to: {model_path}')

    def do_iter(self, iter_id, epoch_id, training):
        if training:
            images, labels = self._next_data_batch(0)
            images, labels = images.to(self.device), labels.to(self.device)

            model = self.model
            criterion_cls = self.criterion
            logits = model(images)
            pred = logits.argmax(1)
            loss = criterion_cls(logits, labels)
            loss.backward()
            self.meters['loss'].update(loss)
            self._update_acc_meter('acc_train_clean', pred, labels)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.iter_count += 1
        else:
            train_images, train_labels = self._next_data_batch(1)
            test_images, test_labels = self._next_data_batch(2)
            train_images, train_labels = train_images.to(self.device), train_labels.to(self.device)
            test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
            with torch.no_grad():
                model = self.model
                pred_train_clean = model(train_images).argmax(1)
                self._update_acc_meter('acc_train_clean', pred_train_clean, train_labels)
                pred_test_clean = model(test_images).argmax(1)
                self._update_acc_meter('acc_test_clean', pred_test_clean, test_labels)

    def log_iter(self, iter_id, epoch_id, training):
        self.update_meters(training)
        # Progress bar
        disp_items = [""]
        for name, info in self.meters_info.items():
            fmt = info['format']
            if fmt is not None:
                value = self.meters[name].get_value()
                disp_items.append(f"{info['abbr']} {value:{fmt}}")
        self.msg = '|'.join(disp_items)
        idx = 0 if training else 1
        self.tqdms[idx].set_postfix_str(self.msg)
        self.tqdms[idx].update(iter_id - self.last_log_iter_id)
        self.last_log_iter_id = iter_id

        # Third-party tools
        if training:
            for name, meter in self.meters.items():
                self.manager.log_metric(name, meter.get_value(),
                                        self.iter_count, epoch_id, split='train')

    def do_epoch(self, epoch_id, training):
        idx = 0 if training else 1
        iters_per_epoch = self.iters_per_epoch[idx]
        self.model.train(training)
        self.setup_logging(training)
        for iter_id in range(iters_per_epoch):
            self.do_iter(iter_id, epoch_id, training)
            if ((iter_id + 1) % self.log_period == 0 or
                                   iter_id == iters_per_epoch - 1):
                self.log_iter(iter_id, epoch_id, training)

    def log_epoch(self, epoch_id, training):
        idx = 0 if training else 1
        self.last_log_iter_id = -1
        self.tqdms[idx].close()
        self.update_meters(training)
        if training:
            self.logger.info(f"train: {self.msg}")
        else:
            self.logger.info(f"test: {self.msg}")
            if (epoch_id + 1) % self.ckpt_period == 0:
                self.save_checkpoint(epoch_id)
            for name, meter in self.meters.items():
                self.manager.log_metric(name, meter.get_value(),
                                   self.iter_count, epoch_id, split='val')

    def setup_logging(self, training):
        # meter_dict = {
        #     'avg': MovingAverageMeter if training else AverageMeter,
        #     'scaler': ScalerMeter,
        # }
        for name, info in self.meters_info.items():
            meter_type = info['type']
            if meter_type == 'avg':
                meter = AverageMeter()
            elif meter_type == 'scaler':
                meter = ScalerMeter()
            else:
                raise NotImplementedError
            self.meters[name] = meter

    def train(self):
        for epoch_id in range(self.num_epochs):
            lrs = self.scheduler.get_last_lr()[0]
            self.logger.info(f'Epoch: {epoch_id}/{self.num_epochs} lr: {lrs:.5f}')
            if (epoch_id + 1) % self.test_period == 0 or \
                    epoch_id == self.num_epochs - 1:
                # True: train
                # False: test
                stages = [True, False]
            else:
                stages = [True]
            for training in stages:
                self._setup_tqdms(training)
                self.do_epoch(epoch_id, training)
                self.log_epoch(epoch_id, training)
            self.scheduler.step()
        self.save_checkpoint(self.num_epochs - 1, 'ckpt')
