import os
from abc import abstractmethod
import time
import torch
import pandas as pd
from numpy import inf
from typing_extensions import final
from modules.abnormality import (
    PATHOLOGY_NAMES,
    PRIMARY_PATHOLOGY_INDICES,
    PRIMARY_PATHOLOGY_NAMES,
)


# class BaseTrainer(object):
#     def __init__(self, model, criterion, metric_ftns, optimizer, args):
#         self.args = args
#
#         # setup GPU device if available, move model into configured device
#         self.device, device_ids = self._prepare_device(args.n_gpu)
#         self.model = model.to(self.device)
#         if len(device_ids) > 1:
#             self.model = torch.nn.DataParallel(model, device_ids=device_ids)
#
#         self.criterion = criterion
#         self.metric_ftns = metric_ftns
#         self.optimizer = optimizer
#
#         self.epochs = self.args.epochs
#         self.save_period = self.args.save_period
#
#         self.mnt_mode = args.monitor_mode
#         self.mnt_metric = 'val_' + args.monitor_metric
#         self.mnt_metric_test = 'test_' + args.monitor_metric
#         assert self.mnt_mode in ['min', 'max']
#
#         self.mnt_best = inf if self.mnt_mode == 'min' else -inf
#         self.early_stop = getattr(self.args, 'early_stop', inf)
#
#         self.start_epoch = 1
#         self.checkpoint_dir = args.save_dir
#
#         if not os.path.exists(self.checkpoint_dir):
#             os.makedirs(self.checkpoint_dir)
#
#         if args.resume is not None:
#             self._resume_checkpoint(args.resume)
#
#         self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
#                               'test': {self.mnt_metric_test: self.mnt_best}}
#
#     @abstractmethod
#     def _train_epoch(self, epoch):
#         raise NotImplementedError
#
#     def train(self):
#         not_improved_count = 0
#         for epoch in range(self.start_epoch, self.epochs + 1):
#             result = self._train_epoch(epoch)
#
#             # save logged informations into log dict
#             log = {'epoch': epoch}
#             log.update(result)
#             self._record_best(log)
#
#             # print logged informations to the screen
#             for key, value in log.items():
#                 print('\t{:15s}: {}'.format(str(key), value))
#
#             # evaluate model performance according to configured metric, save best checkpoint as model_best
#             best = False
#             if self.mnt_mode != 'off':
#                 try:
#                     # check whether model performance improved or not, according to specified metric(mnt_metric)
#                     improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
#                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
#                 except KeyError:
#                     print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
#                         self.mnt_metric))
#                     self.mnt_mode = 'off'
#                     improved = False
#
#                 if improved:
#                     self.mnt_best = log[self.mnt_metric]
#                     not_improved_count = 0
#                     best = True
#                 else:
#                     not_improved_count += 1
#
#                 if not_improved_count > self.early_stop:
#                     print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
#                         self.early_stop))
#                     break
#
#             if epoch % self.save_period == 0:
#                 self._save_checkpoint(epoch, save_best=best)
#         self._print_best()
#         self._print_best_to_file()
#
#     def _print_best_to_file(self):
#         crt_time = time.asctime(time.localtime(time.time()))
#         self.best_recorder['val']['time'] = crt_time
#         self.best_recorder['test']['time'] = crt_time
#         self.best_recorder['val']['seed'] = self.args.seed
#         self.best_recorder['test']['seed'] = self.args.seed
#         self.best_recorder['val']['best_model_from'] = 'val'
#         self.best_recorder['test']['best_model_from'] = 'test'
#
#         if not os.path.exists(self.args.record_dir):
#             os.makedirs(self.args.record_dir)
#         record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
#         if not os.path.exists(record_path):
#             record_table = pd.DataFrame()
#         else:
#             record_table = pd.read_csv(record_path)
#         record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
#         record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
#         record_table.to_csv(record_path, index=False)
#
#     def _prepare_device(self, n_gpu_use):
#         n_gpu = torch.cuda.device_count()
#         if n_gpu_use > 0 and n_gpu == 0:
#             print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
#             n_gpu_use = 0
#         if n_gpu_use > n_gpu:
#             print(
#                 "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
#                     n_gpu_use, n_gpu))
#             n_gpu_use = n_gpu
#         device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
#         list_ids = list(range(n_gpu_use))
#         return device, list_ids
#
#     def _save_checkpoint(self, epoch, save_best=False):
#         state = {
#             'epoch': epoch,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'monitor_best': self.mnt_best
#         }
#
#         # 这样第 1 轮就会存为 checkpoint_epoch_1.pth，第 2 轮就是 checkpoint_epoch_2.pth，互不覆盖
#         filename = os.path.join(self.checkpoint_dir, 'current_checkpoint10.pth')
#         torch.save(state, filename)
#         print("Saving checkpoint: {} ...".format(filename))
#         if save_best:
#             best_path = os.path.join(self.checkpoint_dir, 'model_best10.pth')
#             torch.save(state, best_path)
#             print("Saving current best: model_best9.pth ...")
#
#     def _resume_checkpoint(self, resume_path):
#         resume_path = str(resume_path)
#         print("Loading checkpoint: {} ...".format(resume_path))
#         checkpoint = torch.load(resume_path)
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.mnt_best = checkpoint['monitor_best']
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
#
#     def _record_best(self, log):
#         improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
#             self.mnt_metric]) or \
#                        (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
#         if improved_val:
#             self.best_recorder['val'].update(log)
#
#         improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
#             self.mnt_metric_test]) or \
#                         (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
#                             self.mnt_metric_test])
#         if improved_test:
#             self.best_recorder['test'].update(log)
#
#     def _print_best(self):
#         print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
#         for key, value in self.best_recorder['val'].items():
#             print('\t{:15s}: {}'.format(str(key), value))
#
#         print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
#         for key, value in self.best_recorder['test'].items():
#             print('\t{:15s}: {}'.format(str(key), value))
#
#
# class Trainer(BaseTrainer):
#     def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
#                  test_dataloader):
#         super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
#         self.lr_scheduler = lr_scheduler
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#
#     def _train_epoch(self, epoch):
#
#         train_loss = 0
#         self.model.train()
#         for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
#             images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
#                 self.device)
#
#             #纯swin
#             output = self.model(images, reports_ids, mode='train')
#             #final facol
#             output, image_features, text_features = self.model(
#                 images, reports_ids, mode='train', return_features=True
#             )
#
#             if epoch == 1 and batch_idx < 1:
#                 print(f"[DEBUG] Raw output tensor (first sample): {output[0]}")
#                 print(f"[DEBUG] Model vocabulary size (output.shape[-1]): {output.shape[-1]}")
#                 print(f"[DEBUG] Max word ID in target (reports_ids.max()): {reports_ids.max().item()}")
#                 print(f"[DEBUG] Min word ID in target (reports_ids.min()): {reports_ids.min().item()}")
#             loss = self.criterion(
#                 output,
#                 reports_ids,
#                 reports_masks
#                 #final focal
#                 #image_features=image_features,
#                 #text_features=text_features,
#                 #lambda_cl=0.1
#             )
#
#             train_loss += loss.item()
#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
#             self.optimizer.step()
#
#         log = {'train_loss': train_loss / len(self.train_dataloader)}
#
#         self.model.eval()
#         with torch.no_grad():
#             val_gts, val_res = [], []
#             for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
#                 images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
#                     self.device), reports_masks.to(self.device)
#                 output = self.model(images, mode='sample')
#
#                 tokenizer = self.model.module.tokenizer if isinstance(self.model,
#                                                                       torch.nn.DataParallel) else self.model.tokenizer
#                 reports = tokenizer.decode_batch(output.cpu().numpy())
#                 ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
#
#                 val_res.extend(reports)
#                 val_gts.extend(ground_truths)
#             val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
#                                        {i: [re] for i, re in enumerate(val_res)})
#             log.update(**{'val_' + k: v for k, v in val_met.items()})
#
#         self.model.eval()
#         with torch.no_grad():
#             test_gts, test_res = [], []
#             if self.test_dataloader is not None:
#                 for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
#                     images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
#                         self.device), reports_masks.to(self.device)
#                     output = self.model(images, mode='sample')
#
#                     tokenizer = self.model.module.tokenizer if isinstance(self.model,
#                                                                           torch.nn.DataParallel) else self.model.tokenizer
#                     reports = tokenizer.decode_batch(output.cpu().numpy())
#                     ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
#
#                     test_res.extend(reports)
#                     test_gts.extend(ground_truths)
#             test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
#                                         {i: [re] for i, re in enumerate(test_res)})
#             log.update(**{'test_' + k: v for k, v in test_met.items()})
#
#         self.lr_scheduler.step()
#
#         return log

#寰皟

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.clinical_best = -inf
        self.current_pathology_thresholds = [
            float(getattr(args, 'pathology_threshold', 0.5))
        ] * len(PATHOLOGY_NAMES)
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {
            'val': {self.mnt_metric: self.mnt_best}
        }

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

            clinical_score = log.get('val_clinical_primary_macro_F1')
            if (
                clinical_score is not None
                and clinical_score > self.clinical_best + 1e-12
            ):
                self.clinical_best = clinical_score
                self._save_clinical_checkpoint(epoch, clinical_score)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'clinical_best': self.clinical_best,
            'pathology_names': list(PATHOLOGY_NAMES),
            'primary_pathology_names': list(PRIMARY_PATHOLOGY_NAMES),
            'pathology_thresholds': list(self.current_pathology_thresholds),
        }

        # 杩欐牱绗?1 杞氨浼氬瓨涓?checkpoint_epoch_1.pth锛岀 2 杞氨鏄?checkpoint_epoch_2.pth锛屼簰涓嶈鐩?        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint10.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best10.pth')
            torch.save(state, best_path)
            bleu_path = os.path.join(self.checkpoint_dir, 'model_best_bleu.pth')
            torch.save(state, bleu_path)
            print("Saving current BLEU best: model_best10.pth and model_best_bleu.pth ...")

    def _save_clinical_checkpoint(self, epoch, clinical_score):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'clinical_best': clinical_score,
            'pathology_names': list(PATHOLOGY_NAMES),
            'primary_pathology_names': list(PRIMARY_PATHOLOGY_NAMES),
            'pathology_thresholds': list(self.current_pathology_thresholds),
        }
        clinical_path = os.path.join(
            self.checkpoint_dir, 'model_best_clinical.pth'
        )
        torch.save(state, clinical_path)
        print(
            'Saving current clinical best: model_best_clinical.pth '
            f'(val primary macro-F1={clinical_score:.4f}) ...'
        )

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.clinical_best = checkpoint.get('clinical_best', self.clinical_best)
        saved_thresholds = checkpoint.get('pathology_thresholds')
        if saved_thresholds is not None:
            if len(saved_thresholds) != len(PATHOLOGY_NAMES):
                raise ValueError('checkpoint pathology threshold count is incompatible')
            self.current_pathology_thresholds = [
                float(value) for value in saved_thresholds
            ]
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        #绱ф帴鐫€鍦ㄥ畠涓嬮潰锛屽己琛屾彃鍏ヨ繖涓よ浠ｇ爜锛屾妸鏂板涔犵巼鐏岃繘鍘伙細
        self.optimizer.param_groups[0]['lr'] = self.args.lr_ve
        self.optimizer.param_groups[1]['lr'] = self.args.lr_ed
        if len(self.optimizer.param_groups) > 2:
            self.optimizer.param_groups[2]['lr'] = self.args.lr_pathology
        print(f"  宸叉垚鍔熷皢鎭㈠鍚庣殑浼樺寲鍣ㄥ涔犵巼寮哄埗閲嶇疆涓烘柊鍙傛暟: lr_ve={self.args.lr_ve}, lr_ed={self.args.lr_ed}")
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))
        print(
            'The test set is not inspected during training. Use the export '
            'script once on validation-selected checkpoints.'
        )


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.pathology_pos_weight = torch.tensor(
            train_dataloader.dataset.pathology_pos_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self._original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        self._backbone_frozen = False

    def _set_backbone_warmup(self, enabled):
        if enabled == self._backbone_frozen:
            return
        for name, parameter in self.model.named_parameters():
            is_clinical_head = (
                'view_fusion.' in name
                or 'pathology_classifier.' in name
                or 'pathology_projection.' in name
            )
            if enabled:
                parameter.requires_grad = (
                    self._original_requires_grad[name] and is_clinical_head
                )
            else:
                parameter.requires_grad = self._original_requires_grad[name]
        self._backbone_frozen = enabled
        status = 'frozen' if enabled else 'unfrozen'
        print(f'[CLINICAL WARM-UP] FinalModel backbone is {status}.')

    def _tune_pathology_thresholds(self, probabilities, labels):
        probabilities = torch.cat(probabilities, dim=0)
        labels = torch.cat(labels, dim=0).bool()
        default_threshold = float(
            getattr(self.args, 'pathology_threshold', 0.5)
        )
        search_min = float(getattr(self.args, 'threshold_search_min', 0.1))
        search_max = float(getattr(self.args, 'threshold_search_max', 0.9))
        search_step = float(getattr(self.args, 'threshold_search_step', 0.05))
        if not 0.0 < search_min <= search_max < 1.0:
            raise ValueError('threshold search bounds must satisfy 0 < min <= max < 1')
        if search_step <= 0.0:
            raise ValueError('threshold_search_step must be positive')

        candidates = torch.arange(
            search_min,
            search_max + search_step * 0.5,
            search_step,
            dtype=probabilities.dtype,
        )
        thresholds = torch.full(
            (len(PATHOLOGY_NAMES),),
            default_threshold,
            dtype=probabilities.dtype,
        )
        for class_index in range(len(PATHOLOGY_NAMES)):
            class_labels = labels[:, class_index]
            if class_labels.sum().item() == 0:
                continue
            best_f1 = -1.0
            best_threshold = default_threshold
            for candidate in candidates:
                class_predictions = (
                    probabilities[:, class_index] >= candidate
                )
                tp = (class_predictions & class_labels).sum().float()
                fp = (class_predictions & ~class_labels).sum().float()
                fn = (~class_predictions & class_labels).sum().float()
                precision = tp / (tp + fp).clamp_min(1.0)
                recall = tp / (tp + fn).clamp_min(1.0)
                f1 = (
                    2.0 * precision * recall
                    / (precision + recall).clamp_min(1e-8)
                ).item()
                candidate_value = float(candidate.item())
                if (
                    f1 > best_f1 + 1e-12
                    or (
                        abs(f1 - best_f1) <= 1e-12
                        and abs(candidate_value - default_threshold)
                        < abs(best_threshold - default_threshold)
                    )
                ):
                    best_f1 = f1
                    best_threshold = candidate_value
            thresholds[class_index] = best_threshold
        return thresholds

    @staticmethod
    def _clinical_metrics(probabilities, labels, threshold):
        if isinstance(probabilities, (list, tuple)):
            probabilities = torch.cat(probabilities, dim=0)
        if isinstance(labels, (list, tuple)):
            labels = torch.cat(labels, dim=0)
        labels = labels.bool()
        threshold_tensor = torch.as_tensor(
            threshold, dtype=probabilities.dtype
        )
        if threshold_tensor.numel() == 1:
            threshold_tensor = threshold_tensor.repeat(probabilities.size(1))
        if threshold_tensor.numel() != probabilities.size(1):
            raise ValueError('pathology threshold count does not match logits')
        predictions = probabilities >= threshold_tensor.view(1, -1)

        tp = (predictions & labels).sum(dim=0).float()
        fp = (predictions & ~labels).sum(dim=0).float()
        fn = (~predictions & labels).sum(dim=0).float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        class_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        supported = labels.sum(dim=0) > 0
        macro_f1 = class_f1[supported].mean() if supported.any() else class_f1.mean()

        micro_tp = tp.sum()
        micro_fp = fp.sum()
        micro_fn = fn.sum()
        micro_precision = micro_tp / (micro_tp + micro_fp).clamp_min(1.0)
        micro_recall = micro_tp / (micro_tp + micro_fn).clamp_min(1.0)
        micro_f1 = (
            2.0 * micro_precision * micro_recall
            / (micro_precision + micro_recall).clamp_min(1e-8)
        )

        abnormal_prediction = predictions.any(dim=1)
        abnormal_label = labels.any(dim=1)
        abnormal_tp = (abnormal_prediction & abnormal_label).sum().float()
        abnormal_fp = (abnormal_prediction & ~abnormal_label).sum().float()
        abnormal_fn = (~abnormal_prediction & abnormal_label).sum().float()
        abnormal_precision = abnormal_tp / (
            abnormal_tp + abnormal_fp
        ).clamp_min(1.0)
        abnormal_recall = abnormal_tp / (
            abnormal_tp + abnormal_fn
        ).clamp_min(1.0)
        abnormal_f1 = (
            2.0 * abnormal_precision * abnormal_recall
            / (abnormal_precision + abnormal_recall).clamp_min(1e-8)
        )

        primary_indices = torch.tensor(
            PRIMARY_PATHOLOGY_INDICES, dtype=torch.long
        )
        primary_tp = tp[primary_indices]
        primary_fp = fp[primary_indices]
        primary_fn = fn[primary_indices]
        primary_f1 = class_f1[primary_indices]
        primary_supported = labels[:, primary_indices].sum(dim=0) > 0
        primary_macro_f1 = (
            primary_f1[primary_supported].mean()
            if primary_supported.any()
            else primary_f1.mean()
        )
        primary_micro_tp = primary_tp.sum()
        primary_micro_fp = primary_fp.sum()
        primary_micro_fn = primary_fn.sum()
        primary_micro_precision = primary_micro_tp / (
            primary_micro_tp + primary_micro_fp
        ).clamp_min(1.0)
        primary_micro_recall = primary_micro_tp / (
            primary_micro_tp + primary_micro_fn
        ).clamp_min(1.0)
        primary_micro_f1 = (
            2.0 * primary_micro_precision * primary_micro_recall
            / (primary_micro_precision + primary_micro_recall).clamp_min(1e-8)
        )
        primary_predictions = predictions[:, primary_indices].any(dim=1)
        primary_labels = labels[:, primary_indices].any(dim=1)
        primary_abnormal_tp = (
            primary_predictions & primary_labels
        ).sum().float()
        primary_abnormal_fp = (
            primary_predictions & ~primary_labels
        ).sum().float()
        primary_abnormal_fn = (
            ~primary_predictions & primary_labels
        ).sum().float()
        primary_abnormal_precision = primary_abnormal_tp / (
            primary_abnormal_tp + primary_abnormal_fp
        ).clamp_min(1.0)
        primary_abnormal_recall = primary_abnormal_tp / (
            primary_abnormal_tp + primary_abnormal_fn
        ).clamp_min(1.0)
        primary_abnormal_f1 = (
            2.0 * primary_abnormal_precision * primary_abnormal_recall
            / (
                primary_abnormal_precision + primary_abnormal_recall
            ).clamp_min(1e-8)
        )

        metrics = {
            'clinical_macro_F1': macro_f1.item(),
            'clinical_micro_precision': micro_precision.item(),
            'clinical_micro_recall': micro_recall.item(),
            'clinical_micro_F1': micro_f1.item(),
            'clinical_abnormal_precision': abnormal_precision.item(),
            'clinical_abnormal_recall': abnormal_recall.item(),
            'clinical_abnormal_F1': abnormal_f1.item(),
            'clinical_primary_macro_F1': primary_macro_f1.item(),
            'clinical_primary_micro_precision': primary_micro_precision.item(),
            'clinical_primary_micro_recall': primary_micro_recall.item(),
            'clinical_primary_micro_F1': primary_micro_f1.item(),
            'clinical_primary_abnormal_precision': primary_abnormal_precision.item(),
            'clinical_primary_abnormal_recall': primary_abnormal_recall.item(),
            'clinical_primary_abnormal_F1': primary_abnormal_f1.item(),
        }
        for index, pathology_name in enumerate(PATHOLOGY_NAMES):
            metrics[f'clinical_{pathology_name}_recall'] = recall[index].item()
            metrics[f'clinical_{pathology_name}_F1'] = class_f1[index].item()
            metrics[f'clinical_{pathology_name}_threshold'] = float(
                threshold_tensor[index].item()
            )
        return metrics

    def _train_epoch(self, epoch):

        train_loss = 0
        train_lm_loss = 0
        train_contrastive_loss = 0
        train_pathology_loss = 0
        warmup_enabled = bool(
            getattr(self.args, 'pretrained_final_loaded', False)
            and epoch <= int(getattr(self.args, 'freeze_backbone_epochs', 0))
        )
        self._set_backbone_warmup(warmup_enabled)
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, token_weights,
                        pathology_labels) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks, token_weights, pathology_labels = (
                images.to(self.device),
                reports_ids.to(self.device),
                reports_masks.to(self.device),
                token_weights.to(self.device),
                pathology_labels.to(self.device),
            )

            # One forward pass supplies both report logits and contrastive
            # features. The previous duplicate call doubled compute/memory.
            output, image_features, text_features, pathology_logits = self.model(
                images, reports_ids, mode='train', return_features=True
            )

            if epoch == 1 and batch_idx < 1:
                print(f"[DEBUG] Model output shape: {tuple(output.shape)}")
                print(f"[DEBUG] Max word ID in target (reports_ids.max()): {reports_ids.max().item()}")
                print(f"[DEBUG] Min word ID in target (reports_ids.min()): {reports_ids.min().item()}")
            loss, loss_components = self.criterion(
                output,
                reports_ids,
                reports_masks,
                image_features=image_features,
                text_features=text_features,
                token_weights=token_weights,
                pathology_logits=pathology_logits,
                pathology_labels=pathology_labels,
                pathology_pos_weight=self.pathology_pos_weight,
                return_components=True,
            )

            train_loss += loss.item()
            train_lm_loss += loss_components['lm_loss'].item()
            train_contrastive_loss += loss_components['contrastive_loss'].item()
            train_pathology_loss += loss_components['pathology_loss'].item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

        log = {
            'train_loss': train_loss / len(self.train_dataloader),
            'train_lm_loss': train_lm_loss / len(self.train_dataloader),
            'train_contrastive_loss': train_contrastive_loss / len(self.train_dataloader),
            'train_pathology_loss': train_pathology_loss / len(self.train_dataloader),
            'pathology_token_weight': float(getattr(self.args, 'pathology_token_weight', 1.0)),
            'lambda_pathology': float(getattr(self.args, 'lambda_pathology', 0.2)),
            'weighted_pathology_tokens': int(getattr(self.train_dataloader.dataset, 'weighted_token_count', 0)),
            'train_abnormal_count': int(getattr(self.train_dataloader.dataset, 'abnormal_count', 0)),
            'train_normal_count': int(getattr(self.train_dataloader.dataset, 'normal_count', 0)),
        }

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            val_pathology_probabilities, val_pathology_labels = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks,
                            _token_weights, pathology_labels) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output, pathology_logits = self.model(
                    images, mode='sample', return_pathology=True
                )
                val_pathology_probabilities.append(
                    torch.sigmoid(pathology_logits).cpu()
                )
                val_pathology_labels.append(pathology_labels.cpu())

                tokenizer = self.model.module.tokenizer if isinstance(self.model,
                                                                      torch.nn.DataParallel) else self.model.tokenizer
                reports = tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            tuned_thresholds = self._tune_pathology_thresholds(
                val_pathology_probabilities,
                val_pathology_labels,
            )
            self.current_pathology_thresholds = [
                float(value) for value in tuned_thresholds.tolist()
            ]
            val_clinical = self._clinical_metrics(
                val_pathology_probabilities,
                val_pathology_labels,
                tuned_thresholds,
            )
            log.update(**{'val_' + k: v for k, v in val_clinical.items()})

        print(f"[EPOCH {epoch}] Avg train loss: {train_loss / len(self.train_dataloader):.4f}")
        self.lr_scheduler.step()

        return log

