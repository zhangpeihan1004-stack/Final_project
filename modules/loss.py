"""Language-model, focal and image-text contrastive losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    """Token-normalised focal language-model loss with optional weights."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input,
        target,
        mask,
        gamma=2.0,
        token_weights=None,
        sample_weights=None,
        **kwargs,
    ):
        # The decoder predicts every token after BOS. Align all tensors to the
        # shortest sequence without dropping an additional target token.
        sequence_length = min(input.size(1), target.size(1), mask.size(1))
        input = input[:, :sequence_length]
        target = target[:, :sequence_length]
        mask = mask[:, :sequence_length]

        if token_weights is not None:
            token_weights = token_weights[:, :sequence_length]
        if sequence_length == 0:
            return input.sum() * 0.0

        vocab_size = input.size(-1)
        target_safe = torch.clamp(target, min=0, max=vocab_size - 1)
        log_pt = input.gather(
            2, target_safe.long().unsqueeze(2)
        ).squeeze(2)

        pt = torch.exp(log_pt).clamp(min=0.0, max=1.0)
        focal_weight = (1.0 - pt).pow(float(gamma))
        token_loss = -focal_weight * log_pt

        weighted_mask = mask.to(device=input.device, dtype=input.dtype)

        if token_weights is not None:
            if token_weights.dim() != 2:
                raise ValueError('token_weights must have shape [batch, sequence]')
            if token_weights.shape != weighted_mask.shape:
                raise ValueError(
                    f'token_weights shape {tuple(token_weights.shape)} does not '
                    f'match mask shape {tuple(weighted_mask.shape)}'
                )
            token_weights = token_weights.to(
                device=input.device, dtype=input.dtype
            )
            if torch.any(token_weights < 1.0):
                raise ValueError('token_weights must be greater than or equal to 1.0')
            weighted_mask = weighted_mask * token_weights

        elif sample_weights is not None:
            # Backward compatibility only. New experiments should pass
            # token_weights and keep abnormal_weight at 1.0.
            sample_weights = sample_weights.to(
                device=input.device, dtype=input.dtype
            ).view(-1, 1)
            if sample_weights.size(0) != input.size(0):
                raise ValueError(
                    'sample_weights batch size does not match model output'
                )
            weighted_mask = weighted_mask * sample_weights

        return torch.sum(token_loss * weighted_mask) / weighted_mask.sum().clamp_min(1.0)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError('cl_temperature must be positive')
        self.temperature = float(temperature)

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        logits_per_image = torch.matmul(
            image_features, text_features.t()
        ) / self.temperature
        logits_per_text = logits_per_image.t()

        batch_size = image_features.shape[0]
        labels = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2.0


def compute_loss(
    output,
    reports_ids,
    reports_masks,
    image_features=None,
    text_features=None,
    lambda_cl=0.1,
    focal_gamma=2.0,
    cl_temperature=0.07,
    token_weights=None,
    sample_weights=None,
    pathology_logits=None,
    pathology_labels=None,
    pathology_pos_weight=None,
    lambda_pathology=0.2,
    pathology_loss_type='asymmetric',
    asymmetric_gamma_neg=4.0,
    asymmetric_gamma_pos=1.0,
    asymmetric_clip=0.05,
    return_components=False,
):
    """Compute report, contrastive and pathology-classification losses."""
    shifted_token_weights = None
    if token_weights is not None:
        if token_weights.dim() != 2:
            raise ValueError('token_weights must have shape [batch, sequence]')
        shifted_token_weights = token_weights[:, 1:]

    lm_loss = LanguageModelCriterion()(
        output,
        reports_ids[:, 1:],
        reports_masks[:, 1:],
        gamma=focal_gamma,
        token_weights=shifted_token_weights,
        sample_weights=sample_weights,
    )

    total_loss = lm_loss
    cl_loss = output.new_zeros(())
    pathology_loss = output.new_zeros(())

    if image_features is not None and text_features is not None:
        cl_loss = ContrastiveLoss(temperature=cl_temperature).to(
            image_features.device
        )(image_features, text_features)
        total_loss = total_loss + float(lambda_cl) * cl_loss

    if pathology_logits is not None or pathology_labels is not None:
        if pathology_logits is None or pathology_labels is None:
            raise ValueError(
                'pathology_logits and pathology_labels must be provided together'
            )
        pathology_labels = pathology_labels.to(
            device=pathology_logits.device,
            dtype=pathology_logits.dtype,
        )
        if pathology_logits.shape != pathology_labels.shape:
            raise ValueError(
                f'pathology logits shape {tuple(pathology_logits.shape)} does '
                f'not match labels shape {tuple(pathology_labels.shape)}'
            )
        pos_weight = None
        if pathology_pos_weight is not None:
            pos_weight = pathology_pos_weight.to(
                device=pathology_logits.device,
                dtype=pathology_logits.dtype,
            )
        if pathology_loss_type == 'bce':
            pathology_loss = F.binary_cross_entropy_with_logits(
                pathology_logits,
                pathology_labels,
                pos_weight=pos_weight,
            )
        elif pathology_loss_type == 'asymmetric':
            probability = torch.sigmoid(pathology_logits)
            negative_probability = 1.0 - probability
            clip = float(asymmetric_clip)
            if clip < 0.0 or clip >= 1.0:
                raise ValueError('asymmetric_clip must be in [0, 1)')
            if clip > 0.0:
                negative_probability = (
                    negative_probability + clip
                ).clamp(max=1.0)

            positive_loss = pathology_labels * torch.log(
                probability.clamp_min(1e-8)
            )
            negative_loss = (1.0 - pathology_labels) * torch.log(
                negative_probability.clamp_min(1e-8)
            )
            if pos_weight is not None:
                positive_loss = positive_loss * pos_weight.view(1, -1)

            gamma_neg = float(asymmetric_gamma_neg)
            gamma_pos = float(asymmetric_gamma_pos)
            if gamma_neg < 0.0 or gamma_pos < 0.0:
                raise ValueError('asymmetric focusing exponents must be non-negative')
            if gamma_neg > 0.0 or gamma_pos > 0.0:
                positive_focus = (1.0 - probability).pow(gamma_pos)
                negative_focus = (1.0 - negative_probability).pow(gamma_neg)
                positive_loss = positive_loss * positive_focus
                negative_loss = negative_loss * negative_focus

            pathology_loss = -(positive_loss + negative_loss).mean()
        else:
            raise ValueError(
                f'Unsupported pathology_loss_type: {pathology_loss_type}'
            )
        total_loss = total_loss + float(lambda_pathology) * pathology_loss

    if return_components:
        return total_loss, {
            'lm_loss': lm_loss.detach(),
            'contrastive_loss': cl_loss.detach(),
            'pathology_loss': pathology_loss.detach(),
        }
    return total_loss


