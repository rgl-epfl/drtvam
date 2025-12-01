import torch
import drjit as dr


@dr.wrap(source='drjit', target='torch')
def convert_volume(volume):
    """
    Convert a Dr.Jit tensor volume to a PyTorch tensor.

    Args:
        volume: Dr.Jit tensor representing the volume data.

    Returns:
        PyTorch tensor with the same data.
    """
    return volume ** 2


@dr.wrap(source='torch', target='drjit')
def convert_volume_drjit(volume):
    """
    Convert a PyTorch tensor volume to a Dr.Jit tensor.

    Args:
        volume: PyTorch tensor representing the volume data.

    Returns:
        Dr.Jit tensor with the same data.
    """
    # Convert PyTorch tensor to NumPy array, then to Dr.Jit tensor
    return volume


@dr.wrap(source='drjit', target='torch')
def fft_convolve_3d(volume, kernel):
    """
    Perform 3D FFT-based convolution using rFFT for real-valued inputs.

    Args:
        volume: 3D volume tensor (D, H, W)
        kernel: 3D kernel tensor (Kd, Kh, Kw)

    Returns:
        Convolved volume (same shape as input volume)
    """

    # Perform rFFT on both volume and kernel (optimized for real inputs)
    volume_fft = torch.fft.rfftn(volume, dim=(0, 1, 2))
    kernel_fft = torch.fft.rfftn(kernel, dim=(0, 1, 2))

    # Multiply in frequency domain (convolution theorem)
    result_fft = volume_fft * kernel_fft

    # Inverse rFFT to get result
    result = torch.fft.irfftn(result_fft, volume.shape[0:-1], dim=(0, 1, 2))

    return result

