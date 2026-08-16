import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from .abnormality import PATHOLOGY_NAMES
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                # Conservative augmentation preserves laterality and subtle
                # disease signs in chest radiographs.
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
                transforms.RandomRotation(degrees=3),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        sampler = None
        if (
            split == 'train'
            and bool(getattr(args, 'balanced_disease_sampling', True))
        ):
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(int(getattr(args, 'seed', 9233)))
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(
                    self.dataset.sampling_weights, dtype=torch.double
                ),
                num_samples=len(self.dataset),
                replacement=True,
                generator=sampler_generator,
            )
            print(
                '[BALANCED SAMPLER] enabled for training; '
                f'primary_positive={self.dataset.primary_positive_count}/'
                f'{len(self.dataset)}'
            )

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle if sampler is None else False,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        if sampler is not None:
            self.init_kwargs['sampler'] = sampler
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        (
            images_id,
            images,
            reports_ids,
            reports_masks,
            seq_lengths,
            token_weights,
            pathology_labels,
        ) = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_token_weights = np.ones(
            (len(reports_ids), max_seq_length), dtype=np.float32
        )
        targets_pathology_labels = np.asarray(pathology_labels, dtype=np.float32)
        if targets_pathology_labels.shape != (len(reports_ids), len(PATHOLOGY_NAMES)):
            raise ValueError(
                'pathology label matrix has unexpected shape: '
                f'{targets_pathology_labels.shape}'
            )

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        for i, report_token_weights in enumerate(token_weights):
            if len(report_token_weights) != len(reports_ids[i]):
                raise ValueError(
                    'token_weights length does not match report_ids length '
                    f'for batch item {i}'
                )
            targets_token_weights[i, :len(report_token_weights)] = report_token_weights

        return (
            images_id,
            images,
            torch.LongTensor(targets),
            torch.FloatTensor(targets_masks),
            torch.FloatTensor(targets_token_weights),
            torch.FloatTensor(targets_pathology_labels),
        )


