import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from .abnormality import (
    PATHOLOGY_NAMES,
    PRIMARY_PATHOLOGY_INDICES,
    PRIMARY_PATHOLOGY_NAMES,
    build_pathology_token_weights,
    extract_pathology_labels,
    is_abnormal_report,
    load_abnormal_labels,
)


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.abnormal_weight = float(getattr(args, 'abnormal_weight', 1.0))
        if self.abnormal_weight < 1.0:
            raise ValueError('abnormal_weight must be greater than or equal to 1.0')
        self.pathology_token_weight = float(
            getattr(args, 'pathology_token_weight', 1.0)
        )
        if self.pathology_token_weight < 1.0:
            raise ValueError('pathology_token_weight must be greater than or equal to 1.0')
        if float(getattr(args, 'lambda_pathology', 0.2)) > 0 and self.pathology_token_weight != 1.0:
            print(
                '[WARNING] pathology_token_weight is forced to 1.0 when the '
                'pathology classification branch is enabled.'
            )
            self.pathology_token_weight = 1.0
        if self.abnormal_weight != 1.0:
            print(
                '[WARNING] abnormal_weight is deprecated and ignored. '
                'Only non-negated pathology tokens are weighted.'
            )
        self.abnormal_labels = load_abnormal_labels(
            getattr(args, 'abnormal_labels_path', None)
        )

        self.examples = self.ann[self.split]
        abnormal_count = 0
        weighted_token_count = 0
        pathology_positive_counts = [0] * len(PATHOLOGY_NAMES)
        label_source = 'external label file' if self.abnormal_labels is not None else 'negation-aware report rule'
        for i in range(len(self.examples)):
            report = self.examples[i]['report']
            self.examples[i]['ids'] = tokenizer(report)[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            if self.split == 'train':
                token_weights = build_pathology_token_weights(
                    report,
                    tokenizer,
                    pathology_weight=self.pathology_token_weight,
                )[:self.max_seq_length]
            else:
                token_weights = [1.0] * len(self.examples[i]['ids'])
            if len(token_weights) != len(self.examples[i]['ids']):
                raise ValueError(
                    f'Token-weight alignment failed for study {self.examples[i]["id"]}: '
                    f'{len(token_weights)} weights vs {len(self.examples[i]["ids"])} ids'
                )
            self.examples[i]['token_weights'] = token_weights
            weighted_token_count += sum(weight > 1.0 for weight in token_weights)
            pathology_labels = extract_pathology_labels(report)
            self.examples[i]['pathology_labels'] = pathology_labels
            for label_index, label in enumerate(pathology_labels):
                pathology_positive_counts[label_index] += int(label)
            study_id = str(self.examples[i]['id'])
            if self.abnormal_labels is not None:
                if study_id not in self.abnormal_labels:
                    raise KeyError(f'Missing abnormal label for study {study_id}')
                is_abnormal = self.abnormal_labels[study_id]
            else:
                is_abnormal = is_abnormal_report(self.examples[i]['report'])
            self.examples[i]['is_abnormal'] = bool(is_abnormal)
            abnormal_count += int(is_abnormal)

        self.abnormal_count = abnormal_count
        self.normal_count = len(self.examples) - abnormal_count
        self.abnormal_label_source = label_source
        self.weighted_token_count = weighted_token_count
        self.pathology_names = PATHOLOGY_NAMES
        self.primary_pathology_names = PRIMARY_PATHOLOGY_NAMES
        self.primary_pathology_indices = PRIMARY_PATHOLOGY_INDICES
        self.pathology_positive_counts = pathology_positive_counts
        pos_weight_cap = float(
            getattr(args, 'pathology_pos_weight_cap', 5.0)
        )
        if pos_weight_cap < 1.0:
            raise ValueError('pathology_pos_weight_cap must be at least 1.0')
        self.pathology_pos_weight = [
            min(
                pos_weight_cap,
                max(
                    1.0,
                    (len(self.examples) - positive_count) / max(positive_count, 1),
                ),
            )
            for positive_count in pathology_positive_counts
        ]

        sampling_cap = float(
            getattr(args, 'disease_sampling_max_weight', 3.0)
        )
        if sampling_cap < 1.0:
            raise ValueError('disease_sampling_max_weight must be at least 1.0')
        primary_counts = {
            index: pathology_positive_counts[index]
            for index in PRIMARY_PATHOLOGY_INDICES
        }
        self.sampling_weights = []
        for example in self.examples:
            positive_primary_indices = [
                index for index in PRIMARY_PATHOLOGY_INDICES
                if example['pathology_labels'][index] > 0
            ]
            if not positive_primary_indices:
                sample_weight = 1.0
            else:
                sample_weight = max(
                    (
                        len(self.examples)
                        / max(primary_counts[index], 1)
                    ) ** 0.5
                    for index in positive_primary_indices
                )
                sample_weight = min(sampling_cap, max(1.0, sample_weight))
            self.sampling_weights.append(float(sample_weight))

        self.primary_positive_count = sum(
            any(
                example['pathology_labels'][index] > 0
                for index in PRIMARY_PATHOLOGY_INDICES
            )
            for example in self.examples
        )

        print(
            f'[PATHOLOGY TOKEN WEIGHTING] split={self.split} '
            f'abnormal={self.abnormal_count} normal={self.normal_count} '
            f'token_weight={self.pathology_token_weight if self.split == "train" else 1.0} '
            f'weighted_tokens={self.weighted_token_count} '
            f'label_source={label_source}'
        )
        print(
            f'[PATHOLOGY LABELS] split={self.split} '
            + ' '.join(
                f'{name}={count}'
                for name, count in zip(PATHOLOGY_NAMES, pathology_positive_counts)
            )
        )
        print(
            f'[PRIMARY PATHOLOGIES] split={self.split} '
            f'positive_studies={self.primary_positive_count} '
            f'classes={",".join(PRIMARY_PATHOLOGY_NAMES)} '
            f'sampling_weight_max={max(self.sampling_weights, default=1.0):.3f}'
        )

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        token_weights = example['token_weights']
        pathology_labels = example['pathology_labels']
        sample = (
            image_id,
            image,
            report_ids,
            report_masks,
            seq_length,
            token_weights,
            pathology_labels,
        )
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        token_weights = example['token_weights']
        pathology_labels = example['pathology_labels']
        sample = (
            image_id,
            image,
            report_ids,
            report_masks,
            seq_length,
            token_weights,
            pathology_labels,
        )
        return sample
