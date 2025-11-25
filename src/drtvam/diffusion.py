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
    Perform 3D FFT-based convolution compatible with Dr.Jit AD.

    Args:
        volume: 3D volume tensor (D, H, W)
        kernel: 3D kernel tensor (Kd, Kh, Kw)

    Returns:
        Convolved volume (same shape as input volume)
    """


    # Perform FFT on both volume and kernel
    volume_fft = torch.fft.fftn(volume, dim=(0, 1, 2))
    kernel_fft = torch.fft.fftn(kernel, dim=(0, 1, 2))

    # Multiply in frequency domain (convolution theorem)
    result_fft = volume_fft * kernel_fft

    # Inverse FFT to get result
    result = torch.fft.ifftn(result_fft, dim=(0, 1, 2)).real

    return result#convert_volume_drjit(result)
