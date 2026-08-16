import os
# 璁剧疆 Hugging Face 闀滃儚鍦板潃
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import argparse
import numpy as np
from modules.abnormality import PATHOLOGY_NAMES
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--vocab_path', type=str, default='data/iu_xray/vocab8.pkl', help='path to save/load vocab')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=2, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='swin_base_mfa', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    #R2GEN-BASELINE
    #parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    #TWIN-transformer
    parser.add_argument('--d_vf', type=int, default=512, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the s ample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=40, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=40, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='AdamW', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=3e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--lr_pathology', type=float, default=1e-4,
                        help='learning rate for the pathology head and disease token projection.')
    parser.add_argument('--weight_decay', type=float, default=2e-3, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    # Custom Loss settings (MFA + CL + Focal)
    parser.add_argument('--lambda_cl', type=float, default=0.05, help='the weight for Contrastive Loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='the gamma parameter for Focal Loss.')
    parser.add_argument('--cl_temperature', type=float, default=0.07, help='the temperature for Contrastive Loss.')
    # Legacy report-level weighting is disabled by default. The new training
    # path should weight only non-negated pathology tokens.
    parser.add_argument('--abnormal_weight', type=float, default=1.0,
                        help='legacy report-level abnormal weight; keep at 1.0.')
    parser.add_argument('--pathology_token_weight', type=float, default=1.0,
                        help='legacy pathology-token weight; keep at 1.0 with the classifier.')
    parser.add_argument('--lambda_pathology', type=float, default=0.2,
                        help='weight for the multi-label pathology classification loss.')
    parser.add_argument('--pathology_dropout', type=float, default=0.2,
                        help='dropout used in the pathology classification head.')
    parser.add_argument('--view_fusion_dropout', type=float, default=0.2,
                        help='dropout in the frontal/lateral interaction fusion block.')
    parser.add_argument('--pathology_threshold', type=float, default=0.5,
                        help='validation/test threshold for pathology probabilities.')
    parser.add_argument('--threshold_search_min', type=float, default=0.1,
                        help='minimum validation threshold considered per pathology.')
    parser.add_argument('--threshold_search_max', type=float, default=0.9,
                        help='maximum validation threshold considered per pathology.')
    parser.add_argument('--threshold_search_step', type=float, default=0.05,
                        help='validation threshold-search step per pathology.')
    parser.add_argument('--balanced_disease_sampling',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='oversample studies containing one of the six primary findings.')
    parser.add_argument('--disease_sampling_max_weight', type=float, default=3.0,
                        help='maximum study sampling weight for primary findings.')
    parser.add_argument('--pathology_pos_weight_cap', type=float, default=5.0,
                        help='maximum positive-class loss weight.')
    parser.add_argument('--pathology_loss_type', type=str, default='asymmetric',
                        choices=['bce', 'asymmetric'],
                        help='multi-label pathology loss.')
    parser.add_argument('--asymmetric_gamma_neg', type=float, default=4.0,
                        help='negative focusing exponent for asymmetric loss.')
    parser.add_argument('--asymmetric_gamma_pos', type=float, default=1.0,
                        help='positive focusing exponent for asymmetric loss.')
    parser.add_argument('--asymmetric_clip', type=float, default=0.05,
                        help='negative probability margin for asymmetric loss.')
    parser.add_argument('--pretrained_final_checkpoint', type=str, default=None,
                        help='original FinalModel checkpoint used to initialise the clinical model.')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3,
                        help='head warm-up epochs when a pretrained FinalModel is loaded.')
    parser.add_argument('--abnormal_labels_path', type=str, default=None,
                        help='optional JSON study-level abnormal labels; report rules are used when omitted.')
    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=4, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    args.num_pathologies = len(PATHOLOGY_NAMES)
    return args


def load_pretrained_final_model(model, checkpoint_path):
    """Load the original FinalModel while leaving new clinical layers fresh."""
    if not checkpoint_path:
        print('[CLINICAL INIT] No pretrained FinalModel checkpoint supplied.')
        return False
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Pretrained FinalModel checkpoint not found: {checkpoint_path}'
        )

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith('module.') or key.startswith('_orig_mod.'):
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('_orig_mod.'):
                key = key[len('_orig_mod.'):]
        cleaned_state_dict[key] = value

    incompatible = model.load_state_dict(cleaned_state_dict, strict=False)
    allowed_missing_prefixes = (
        'view_fusion.',
        'pathology_classifier.',
        'pathology_projection.',
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing:
        raise RuntimeError(
            'Original FinalModel checkpoint is incompatible. Unexpected '
            f'missing keys: {unexpected_missing[:20]}'
        )
    print(
        f'[CLINICAL INIT] Loaded original FinalModel: {checkpoint_path}; '
        f'new clinical parameters={len(incompatible.missing_keys)}; '
        f'unexpected checkpoint keys={len(incompatible.unexpected_keys)}'
    )
    return True


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)


    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)
    args.pretrained_final_loaded = load_pretrained_final_model(
        model, args.pretrained_final_checkpoint
    )

    # get function handles of loss and metrics
    # 鍒╃敤 lambda 鍑芥暟锛屾妸鍛戒护琛屼紶杩涙潵鐨?args 鍔ㄦ€佺粦瀹氱粰 compute_loss

    # Prefer pathology-token weights, while remaining compatible with the
    # current report-level loader until the remaining modules are migrated.
    def criterion_wrapper(output, reports_ids, reports_masks, **kwargs):
        weighting_kwargs = {}
        token_weights = kwargs.get('token_weights')
        sample_weights = kwargs.get('sample_weights')

        if token_weights is not None:
            weighting_kwargs['token_weights'] = token_weights
        elif sample_weights is not None:
            # Backward compatibility with the current report-level loader.
            weighting_kwargs['sample_weights'] = sample_weights

        return compute_loss(
            output,
            reports_ids,
            reports_masks,
            image_features=kwargs.get('image_features'),
            text_features=kwargs.get('text_features'),
            lambda_cl=args.lambda_cl,
            focal_gamma=args.focal_gamma,
            cl_temperature=args.cl_temperature,
            pathology_logits=kwargs.get('pathology_logits'),
            pathology_labels=kwargs.get('pathology_labels'),
            pathology_pos_weight=kwargs.get('pathology_pos_weight'),
            lambda_pathology=args.lambda_pathology,
            pathology_loss_type=args.pathology_loss_type,
            asymmetric_gamma_neg=args.asymmetric_gamma_neg,
            asymmetric_gamma_pos=args.asymmetric_gamma_pos,
            asymmetric_clip=args.asymmetric_clip,
            return_components=bool(kwargs.get('return_components', False)),
            **weighting_kwargs,
        )

    criterion = criterion_wrapper
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    # The test set is deliberately not loaded during training. It is evaluated
    # once, after model selection, by export_test_results.py.
    trainer = Trainer(
        model, criterion, metrics, optimizer, args, lr_scheduler,
        train_dataloader, val_dataloader, None
    )
    trainer.train()


if __name__ == '__main__':
    main()


