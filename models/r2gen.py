import numpy as np
import torch
import torch.nn as nn

from modules.encoder_decoder import EncoderDecoder
from modules.visual_extractor import VisualExtractor


class R2GenModel(nn.Module):
    """FinalModel with a disease-recognition branch and disease token."""

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        if hasattr(tokenizer, "get_vocab_size"):
            base_vocab_size = tokenizer.get_vocab_size()
        elif hasattr(tokenizer, "token2idx"):
            base_vocab_size = len(tokenizer.token2idx)
        else:
            base_vocab_size = len(tokenizer.idx2word)

        safe_vocab_size = base_vocab_size + 100
        print(f"[DEBUG] tokenizer vocabulary size: {base_vocab_size}")
        print(f"[DEBUG] contrastive embedding capacity: {safe_vocab_size}")
        self.cl_text_embed = nn.Embedding(safe_vocab_size, args.d_model)

        self.num_pathologies = int(args.num_pathologies)
        fusion_dim = int(args.d_vf)
        self.view_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.GELU(),
            nn.Dropout(float(getattr(args, "view_fusion_dropout", 0.2))),
            nn.Linear(fusion_dim, fusion_dim),
        )
        # Start from the original mean-fusion behaviour, then learn a residual
        # interaction among frontal, lateral, difference and agreement cues.
        nn.init.zeros_(self.view_fusion[-1].weight)
        nn.init.zeros_(self.view_fusion[-1].bias)
        self.pathology_classifier = nn.Sequential(
            nn.LayerNorm(args.d_vf),
            nn.Dropout(float(getattr(args, "pathology_dropout", 0.2))),
            nn.Linear(args.d_vf, self.num_pathologies),
        )
        self.pathology_projection = nn.Sequential(
            nn.Linear(self.num_pathologies, args.d_vf),
            nn.Tanh(),
        )
        nn.init.zeros_(self.pathology_projection[0].weight)
        nn.init.zeros_(self.pathology_projection[0].bias)

        if args.dataset_name == "iu_xray":
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        parameters = filter(lambda parameter: parameter.requires_grad, self.parameters())
        count = sum(np.prod(parameter.size()) for parameter in parameters)
        return super().__str__() + f"\nTrainable parameters: {count}"

    def _add_pathology_token(self, att_feats, image_features):
        pathology_logits = self.pathology_classifier(image_features)
        # BCE trains the classifier. Detaching here prevents the language loss
        # from distorting calibrated disease probabilities, while the disease
        # projection and decoder still learn how to use those probabilities.
        pathology_probabilities = torch.sigmoid(pathology_logits).detach()
        pathology_token = self.pathology_projection(
            pathology_probabilities
        ).unsqueeze(1)
        conditioned_att_feats = torch.cat((att_feats, pathology_token), dim=1)
        return conditioned_att_feats, pathology_logits

    def _masked_text_features(self, targets):
        safe_targets = torch.clamp(
            targets,
            min=0,
            max=self.cl_text_embed.num_embeddings - 1,
        )
        text_embeds = self.cl_text_embed(safe_targets)
        valid_mask = targets.ne(0).unsqueeze(-1).to(text_embeds.dtype)
        return (text_embeds * valid_mask).sum(dim=1) / valid_mask.sum(
            dim=1
        ).clamp_min(1.0)

    def _fuse_iu_views(self, frontal_features, lateral_features):
        interaction_features = torch.cat(
            (
                frontal_features,
                lateral_features,
                torch.abs(frontal_features - lateral_features),
                frontal_features * lateral_features,
            ),
            dim=1,
        )
        residual = self.view_fusion(interaction_features)
        return (frontal_features + lateral_features) / 2.0 + residual

    def forward_iu_xray(
        self,
        images,
        targets=None,
        mode="train",
        return_features=False,
        return_pathology=False,
    ):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        image_features = self._fuse_iu_views(fc_feats_0, fc_feats_1)
        att_feats, pathology_logits = self._add_pathology_token(
            att_feats, image_features
        )

        if mode == "train":
            output = self.encoder_decoder(
                fc_feats, att_feats, targets, mode="forward"
            )
            if return_features and targets is not None:
                text_features = self._masked_text_features(targets)
                return output, image_features, text_features, pathology_logits
        elif mode == "sample":
            output, _ = self.encoder_decoder(
                fc_feats, att_feats, mode="sample"
            )
            if return_pathology:
                return output, pathology_logits
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return output

    def forward_mimic_cxr(
        self,
        images,
        targets=None,
        mode="train",
        return_features=False,
        return_pathology=False,
    ):
        att_feats, fc_feats = self.visual_extractor(images)
        image_features = fc_feats
        att_feats, pathology_logits = self._add_pathology_token(
            att_feats, image_features
        )

        if mode == "train":
            output = self.encoder_decoder(
                fc_feats, att_feats, targets, mode="forward"
            )
            if return_features and targets is not None:
                text_features = self._masked_text_features(targets)
                return output, image_features, text_features, pathology_logits
        elif mode == "sample":
            output, _ = self.encoder_decoder(
                fc_feats, att_feats, mode="sample"
            )
            if return_pathology:
                return output, pathology_logits
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return output


