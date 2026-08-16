import torch


def build_optimizer(args, model):
    ve_params = list(model.visual_extractor.parameters())
    pathology_params = (
        list(model.view_fusion.parameters())
        + list(model.pathology_classifier.parameters())
        + list(model.pathology_projection.parameters())
    )
    excluded_ids = {id(parameter) for parameter in ve_params + pathology_params}
    ed_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in excluded_ids
    ]
    optimizer = getattr(torch.optim, args.optim)(
        [
            {'params': ve_params, 'lr': args.lr_ve},
            {'params': ed_params, 'lr': args.lr_ed},
            {'params': pathology_params, 'lr': args.lr_pathology},
        ],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler


