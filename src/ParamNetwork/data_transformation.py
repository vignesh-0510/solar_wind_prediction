import numpy as np
import torch


def signed_sqrt_transform(array):
    return np.sign(array) * (np.abs(array)**0.5)

def signed_pow_transform(array, power):
    return np.sign(array) * (np.abs(array)**power)

def signed_log_transform(array):
    return np.sign(array) * np.log(np.abs(array)+1)

def arcsinh_transform(array: np.ndarray, scale: torch.Tensor, eps: float = 1e-16):
    """
    Arcsinh transform with component-wise scale.

    x:     tensor of shape (B, C, H, W) or similar
    scale: tensor of shape (C,) or broadcastable to x
    """
    scale = np.asarray(scale, dtype=array.dtype)

    if scale.ndim == 1:
        if array.ndim == 4:      # (N, C, H, W)
            scale = scale.reshape(1, -1, 1, 1)
        elif array.ndim == 3:    # (C, H, W)
            scale = scale.reshape(-1, 1, 1)

    return np.arcsinh(array / (scale + eps))
 
def sinh_arcsinh_transform(array, epsilon=0.0, delta=1.0):
    """
    Sinh-arcsinh transformation.

    epsilon controls skewness.
    delta controls tail weight.

    epsilon = 0 and delta = 1 gives identity transform.
    delta > 1 reduces tail heaviness.
    delta < 1 increases tail heaviness.
    """
    return np.sinh(delta * np.arcsinh(array) - epsilon)

def signed_transform(array, transform='sqrt', power=None, epsilon=0.0, delta=1.0, scale=None):
    if transform == 'sqrt':
        return signed_sqrt_transform(array)
    elif transform == 'log':
        return signed_log_transform(array)
    elif transform == 'pow' and power is not None:
        return signed_pow_transform(array, power)
    elif transform == 'arcsinh':
        return arcsinh_transform(array, scale=scale)
    elif transform == 'sinh_arcsinh':
        return sinh_arcsinh_transform(array, epsilon=epsilon, delta=delta)
    else:
        raise ValueError("Unsupported transform type or missing power for 'pow' transform.")

def signed_power_inverse_transform(array, power):
    return torch.sign(array) * (torch.abs(array)**power)

def signed_exp_inverse_transform(array):
    return torch.sign(array) * (torch.exp(torch.abs(array)) - 1)
# def arcsinh_inverse_transform(array):
#     return torch.sinh(array)

def arcsinh_inverse_transform(x_tf: torch.Tensor, scale: torch.Tensor):
    scale = scale.to(x_tf.device, x_tf.dtype)

    if scale.ndim == 1:
        shape = [1, -1] + [1] * (x_tf.ndim - 2)
        scale = scale.view(*shape)

    return torch.sinh(x_tf) * scale

def sinh_arcsinh_inverse_transform(array, epsilon=0.0, delta=1.0):
    return torch.sinh((torch.asinh(array) + epsilon) / delta)

def signed_inverse_transform(array, transform='square', power=None, epsilon=0.0, delta=1.0, scale=None):
    
    if transform == 'square':
        return signed_power_inverse_transform(array, power=2)
    elif transform == 'exp':
        return signed_exp_inverse_transform(array)  # Subtract 1 to reverse the +1 in the log transform
    elif transform == 'pow' and power is not None:
        return signed_power_inverse_transform(array, power=power)
    elif transform == 'sinh':
        return arcsinh_inverse_transform(array, scale=scale)
    elif transform == 'sinh_arcsinh':
        return sinh_arcsinh_inverse_transform(array, epsilon=epsilon, delta=delta)
    else:
        raise ValueError("Unsupported inverse transform type or missing power for 'pow' inverse transform.")

TF_INV_TF_DICT = {
    'sqrt': 'square',
    'log': 'exp',
    'pow': 'pow',
    'arcsinh': 'sinh',
    'sinh_arcsinh': 'sinh_arcsinh'
}

def get_inverse_transform(transform):
    return TF_INV_TF_DICT.get(transform, None)

def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array, axis=(0,2,3))
        max_ = np.max(array, axis=(0,2,3))
    array = (array - min_[:,None, None]) / (max_[:,None, None] - min_[:,None, None] + 1e-9)
    return array, min_, max_

def min_max_denormalize(array, min_, max_):
    '''
    Denormalize the data using the provided min and max values.
    :param array: Normalized array of shape (N, 11, H, W)
    :param min_: Minimum values for each channel (shape: (11,))
    :param max_: Maximum values for each channel (shape: (11,))
    '''
    array = array * (max_[:, None, None] - min_[:, None, None] + 1e-9) + min_[:, None, None]
    return array

def standardize(array, mean_=None, std_=None, eps=1e-12):
    """
    Standardize data per channel.

    Parameters:
    - array: np.ndarray of shape (N, C, H, W)
    - mean_: optional np.ndarray of shape (C,)
    - std_: optional np.ndarray of shape (C,)

    Returns:
    - standardized array
    - mean_
    - std_
    """
    if mean_ is None or std_ is None:
        mean_ = np.mean(array, axis=(0, 2, 3))
        std_ = np.std(array, axis=(0, 2, 3))

    std_ = np.maximum(std_, eps)

    array = (array - mean_[None, :, None, None]) / std_[None, :, None, None]

    return array, mean_, std_


def destandardize(array, mean_, std_):
    """
    Reverse standardization.

    Parameters:
    - array: standardized np.ndarray of shape (N, C, H, W)
    - mean_: np.ndarray of shape (C,)
    - std_: np.ndarray of shape (C,)

    Returns:
    - array in transformed physical scale before standardization
    """
    array = array * std_[None, :, None, None] + mean_[None, :, None, None]
    return array

def get_transform_scale(array, method="std", q=95, eps=1e-8):
    """
    array: numpy array or torch tensor with shape (N, C, H, W)
    returns: torch tensor of shape (C,)
    """
    if method is None:
        return torch.ones(array.shape[1], dtype=torch.float32)

    if isinstance(array, np.ndarray):
        if method == "std":
            scale = np.std(array, axis=(0, 2, 3))
        elif method == "percentile":
            scale = np.percentile(np.abs(array), q, axis=(0, 2, 3))
        elif method == "mean":
            scale = np.mean(np.abs(array), axis=(0, 2, 3))
        else:
            raise ValueError(f"Unknown scale method: {method}")

        scale = np.maximum(scale, eps)
        return torch.tensor(scale, dtype=torch.float32)

    elif isinstance(array, torch.Tensor):
        if method == "std":
            scale = torch.std(array, dim=(0, 2, 3))
        elif method == "percentile":
            scale = torch.quantile(
                torch.abs(array).reshape(array.shape[0], array.shape[1], -1),
                q / 100.0,
                dim=(0, 2),
            )
        elif method == "mean":
            scale = torch.mean(torch.abs(array), dim=(0, 2, 3))
        else:
            raise ValueError(f"Unknown scale method: {method}")

        return torch.clamp(scale, min=eps).float()

    else:
        raise TypeError(f"Expected numpy array or torch tensor, got {type(array)}")