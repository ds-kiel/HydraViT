""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Union

import torch
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

__all__ = ['clean_state_dict', 'load_state_dict', 'load_checkpoint', 'remap_state_dict', 'resume_checkpoint']


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
) -> Dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = False,
        filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # HydraViT: Load a smaller pretrained model into a larger model
    
    first_key = list(state_dict.keys())[0]
    first_key_state = state_dict[first_key]
    first_key_model = model.state_dict()[first_key]
    is_model_larger = first_key_state.reshape(-1).shape[0] < first_key_model.reshape(-1).shape[0]
    
    if not is_model_larger:
        for w in model.state_dict().keys():
            weight_shape = model.state_dict()[w].shape
            if len(weight_shape) == 1:
                if 'qkv' in w:
                    qkv_subtensor = int(weight_shape[0]/3)
                    selected_index = torch.tensor([i for i in range(2304)]).reshape(3, -1)[:,:qkv_subtensor].reshape(-1)
                    model.state_dict()[w][:] = state_dict[w][selected_index]
                else:
                    model.state_dict()[w][:] = state_dict[w][0:weight_shape[0]]
            elif len(weight_shape) == 2:
                if 'qkv' in w:
                    qkv_subtensor = int(weight_shape[0]/3)
                    selected_index = torch.tensor([i for i in range(2304)]).reshape(3, -1)[:,:qkv_subtensor].reshape(-1)
                    try:
                        model.state_dict()[w][:] = state_dict[w][selected_index, 0:qkv_subtensor]
                    except:
                        model.state_dict()[w][:] = state_dict[w][selected_index, :]
                else:
                    model.state_dict()[w][:] = state_dict[w][0:weight_shape[0], 0:weight_shape[1]]
        
            elif len(weight_shape) == 3:
                model.state_dict()[w][:] = state_dict[w][0:weight_shape[0], 0:weight_shape[1], 0:weight_shape[2]]
        
            elif len(weight_shape) == 4:
                model.state_dict()[w][:] = state_dict[w][0:weight_shape[0], 0:weight_shape[1], 0:weight_shape[2], 0:weight_shape[3]]
    else:
        print("Model is larger than the checkpoint")
        for tensor_s in state_dict.keys():
            small_tensor_shape = state_dict[tensor_s].shape
            if len(small_tensor_shape) == 1:
                if 'qkv' in tensor_s:
                    s_size = small_tensor_shape[0] // 3
                    b_size = model.state_dict()[tensor_s].shape[0] // 3
                    model.state_dict()[tensor_s][0:s_size] = state_dict[tensor_s][0:s_size]
                    model.state_dict()[tensor_s][b_size:b_size+s_size] = state_dict[tensor_s][s_size:2*s_size]
                    model.state_dict()[tensor_s][2*b_size:2*b_size+s_size] = state_dict[tensor_s][2*s_size:3*s_size]
                else:
                    model.state_dict()[tensor_s][0:small_tensor_shape[0]] = state_dict[tensor_s]
        
            if len(small_tensor_shape) == 2:
                if 'qkv' in tensor_s:
                    s_size = small_tensor_shape[0] // 3
                    b_size = model.state_dict()[tensor_s].shape[0] // 3
                    model.state_dict()[tensor_s][0:s_size, 0:small_tensor_shape[1]] = state_dict[tensor_s][0:s_size, 0:small_tensor_shape[1]]
                    model.state_dict()[tensor_s][b_size:b_size+s_size, 0:small_tensor_shape[1]] = state_dict[tensor_s][s_size:2*s_size, 0:small_tensor_shape[1]]
                    model.state_dict()[tensor_s][2*b_size:2*b_size+s_size, 0:small_tensor_shape[1]] = state_dict[tensor_s][2*s_size:3*s_size, 0:small_tensor_shape[1]]
                else:
                    model.state_dict()[tensor_s][0:small_tensor_shape[0], 0:small_tensor_shape[1]] = state_dict[tensor_s]
            if len(small_tensor_shape) == 3:
                model.state_dict()[tensor_s][0:small_tensor_shape[0], 0:small_tensor_shape[1], 0:small_tensor_shape[2]] = state_dict[tensor_s]
                    
            if len(small_tensor_shape) == 4:
                model.state_dict()[tensor_s][0:small_tensor_shape[0], 0:small_tensor_shape[1], 0:small_tensor_shape[2], 0:small_tensor_shape[3]] = state_dict[tensor_s]
            
    # --------------------------------------------------------------------------------------------------------------------------------------------



def remap_state_dict(
        state_dict: Dict[str, Any],
        model: torch.nn.Module,
        allow_reshape: bool = True
):
    """ remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False,  f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict


def resume_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer = None,
        loss_scaler: Any = None,
        log_info: bool = True,
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

                if log_info:
                    _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


