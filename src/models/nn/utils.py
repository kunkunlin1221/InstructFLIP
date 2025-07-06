import torch.nn as nn


def init_params(model):
    for m in model.parameters():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


def freeze_model(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = False


def melt_model(m: nn.Module):
    for param in m.parameters():
        param.requires_grad = True


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


# def replace_layer_with_lora(model, rank=16, lora_alpha=32, lora_dropout=0.):
#     if isinstance(model, nn.Linear):
#         lora_model = lora.Linear(
#             in_features=model.in_features,
#             out_features=model.out_features,
#             bias=model.bias is not None,
#             r=rank,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#         )
#         lora_model.weight.data = model.weight.data
#         if model.bias is not None:
#             lora_model.bias.data = model.bias.data
#         lora_model = lora_model.to(dtype=model.weight.dtype, device=model.weight.device)
#     elif isinstance(model, nn.Conv2d):
#         lora_model = lora.Conv2d(
#             in_channels=model.in_channels,
#             out_channels=model.out_channels,
#             kernel_size=model.kernel_size[0],
#             stride=model.stride[0],
#             padding=model.padding[0],
#             bias=model.bias is not None,
#             r=rank,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#         )
#         lora_model.conv.weight.data = model.weight.data
#         if model.bias is not None:
#             lora_model.conv.bias.data = model.bias.data
#         lora_model = lora_model.to(dtype=model.weight.dtype, device=model.weight.device)
#     elif isinstance(model, nn.Embedding):
#         lora_model = lora.Embedding(
#             num_embeddings=model.num_embeddings,
#             embedding_dim=model.embedding_dim,
#             r=rank,
#             lora_alpha=lora_alpha,
#         )
#         lora_model.weight.data = model.weight.data
#         lora_model = lora_model.to(dtype=model.weight.dtype, device=model.weight.device)
#     else:
#         lora_model = model
#     return lora_model


# def replace_layers_with_lora(model, rank=16, lora_alpha=32, lora_dropout=0.):
#     for name, m in model.named_children():
#         if has_children(m):
#             replace_layers_with_lora(m, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
#         else:
#             replaced = replace_layer_with_lora(m, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
#             setattr(model, name, replaced)
#     return model
