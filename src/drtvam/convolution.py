import drjit as dr
import mitsuba as mi

def make_drjit_conv(spacingz, spacingy, spacingx, diffusion_D, delta_t,
                    radiusz=10, radiusy=10, radiusx=10, boundary='reflect'):

    def make_filter(spacing: float, radius: int):
        indices = dr.arange(mi.Float, -radius, radius + 1)
        coords  = indices * spacing
        diffusion_kernel = dr.exp(-coords**2 / (4 * diffusion_D * delta_t))
        norm    = float(dr.sum(diffusion_kernel).item())
        def f(i):
            if abs(i) > radius:
                return 0.0
            x = i * spacing
            diffusion_kernel = dr.exp(-x**2 / (4 * diffusion_D * delta_t))
            return diffusion_kernel / norm
        return f


    # drjit.conv needs 1D separable kernels called
    # with integer indices based around -2,-1,0,1,2
    filter_z = make_filter(spacingz, radiusz)
    filter_y = make_filter(spacingy, radiusy)
    filter_x = make_filter(spacingx, radiusx)

    # this function is called from within the physical model
    # volumes are stored as (z, y, x, channels)
    # the default 'reflect' boundary means no flux through the grid boundary
    def conv(r):
        r = dr.convolve(r, filter=filter_z, filter_radius=radiusz, axis=0, boundary=boundary)
        r = dr.convolve(r, filter=filter_y, filter_radius=radiusy, axis=1, boundary=boundary)
        r = dr.convolve(r, filter=filter_x, filter_radius=radiusx, axis=2, boundary=boundary)
        return r

    return conv


#@dr.wrap(source='drjit', target='torch')
#def convert_volume(volume):
#    """
#    Convert a Dr.Jit tensor volume to a PyTorch tensor.
#
#    Args:
#        volume: Dr.Jit tensor representing the volume data.
#
#    Returns:
#        PyTorch tensor with the same data.
#    """
#    return volume
#
#
#@dr.wrap(source='torch', target='drjit')
#def convert_volume_drjit(volume):
#    """
#    Convert a PyTorch tensor volume to a Dr.Jit tensor.
#
#    Args:
#        volume: PyTorch tensor representing the volume data.
#
#    Returns:
#        Dr.Jit tensor with the same data.
#    """
#    # Convert PyTorch tensor to NumPy array, then to Dr.Jit tensor
#    return volume
#
#
#@dr.wrap(source='drjit', target='torch')
#def fft_convolve_3d(volume, kernel):
#    """
#    Perform 3D FFT-based convolution using rFFT for real-valued inputs.
#
#    Args:
#        volume: 3D volume tensor (D, H, W)
#        kernel: 3D kernel tensor (Kd, Kh, Kw)
#
#    Returns:
#        Convolved volume (same shape as input volume)
#    """
#
#    # Perform rFFT on both volume and kernel (optimized for real inputs)
#    volume_fft = torch.fft.rfftn(volume, dim=(0, 1, 2))
#    kernel_fft = torch.fft.rfftn(kernel, dim=(0, 1, 2))
#
#    # Multiply in frequency domain (convolution theorem)
#    result_fft = volume_fft * kernel_fft
#
#    # Inverse rFFT to get result
#    result = torch.fft.irfftn(result_fft, volume.shape[0:-1], dim=(0, 1, 2))
#
#    return result
#
