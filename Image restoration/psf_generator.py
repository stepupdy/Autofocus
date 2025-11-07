"""
Image restoration/psf_generator.py

Generate an approximate 3D PSF (Gaussian approximation) for fluorescence microscopy,
and extract a 2D slice at a requested defocus depth for 2D convolution or deconvolution.

API:
- generate_3d_gaussian_psf(nx, ny, nz, pixel_size_xy_um, z_spacing_um,
                           wavelength_nm, NA, refractive_index=1.33)
    -> returns psf (nz, ny, nx) normalized so sum == 1

- extract_psf_slice(psf, z_spacing_um, z_offset_um)
    -> returns a single 2D slice (ny, nx) corresponding to the PSF at z = z_offset_um
       (z_offset_um = 0 means the in-focus slice at center)

- save_psf_tiff(psf, out_path)
    -> saves PSF as a multi-page TIFF (z, y, x)
"""
import numpy as np
import tifffile as tiff
from typing import Tuple

def _fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def generate_3d_gaussian_psf(nx: int,
                             ny: int,
                             nz: int,
                             pixel_size_xy_um: float,
                             z_spacing_um: float,
                             wavelength_nm: float,
                             NA: float,
                             refractive_index: float = 1.33) -> np.ndarray:
    """
    Generate approximate 3D PSF using separable Gaussian approximation.
    Output shape: (nz, ny, nx)
    """
    wavelength_um = float(wavelength_nm) * 1e-3  # nm -> um

    # approximations
    lateral_fwhm_um = 0.51 * wavelength_um / float(NA)
    axial_fwhm_um = 2.0 * refractive_index * wavelength_um / (float(NA) ** 2)

    sigma_xy_um = _fwhm_to_sigma(lateral_fwhm_um)
    sigma_z_um = _fwhm_to_sigma(axial_fwhm_um)

    sigma_x_px = sigma_xy_um / pixel_size_xy_um
    sigma_y_px = sigma_xy_um / pixel_size_xy_um
    sigma_z_px = sigma_z_um / z_spacing_um

    x = (np.arange(nx) - (nx - 1) / 2.0)
    y = (np.arange(ny) - (ny - 1) / 2.0)
    z = (np.arange(nz) - (nz - 1) / 2.0)

    xx, yy = np.meshgrid(x, y, indexing='xy')
    rr2 = (xx ** 2) / (sigma_x_px ** 2) + (yy ** 2) / (sigma_y_px ** 2)
    lateral2d = np.exp(-0.5 * rr2)

    zz2 = (z ** 2) / (sigma_z_px ** 2)
    axial1d = np.exp(-0.5 * zz2)

    psf = np.zeros((nz, ny, nx), dtype=np.float32)
    for iz in range(nz):
        psf[iz, :, :] = lateral2d * axial1d[iz]

    s = psf.sum()
    if s != 0:
        psf /= s
    return psf

def extract_psf_slice(psf: np.ndarray, z_spacing_um: float, z_offset_um: float) -> np.ndarray:
    """
    Extract a 2D PSF slice that corresponds to the requested z offset (in um)
    relative to PSF center. If exact slice index falls between planes, the function
    returns a linear interpolation between nearest slices.
    """
    nz = psf.shape[0]
    center_idx = (nz - 1) / 2.0
    # target index relative to center:
    idx = center_idx + (z_offset_um / z_spacing_um)
    if idx <= 0:
        return psf[0].copy()
    if idx >= nz - 1:
        return psf[-1].copy()
    low = int(np.floor(idx))
    high = int(np.ceil(idx))
    w = idx - low
    slice2d = (1 - w) * psf[low] + w * psf[high]
    # normalize
    s = slice2d.sum()
    if s != 0:
        slice2d = slice2d / s
    return slice2d.astype(np.float32)

def save_psf_tiff(psf: np.ndarray, out_path: str, overwrite: bool = True):
    if not overwrite:
        import os
        if os.path.exists(out_path):
            raise FileExistsError(out_path)
    tiff.imwrite(out_path, psf.astype(np.float32))

if __name__ == '__main__':
    # quick demo parameters
    psf = generate_3d_gaussian_psf(nx=65, ny=65, nz=41,
                                   pixel_size_xy_um=0.065, z_spacing_um=0.2,
                                   wavelength_nm=520, NA=0.8, refractive_index=1.33)
    save_psf_tiff(psf, 'psf_example_3d.tif')
    print('Saved psf_example_3d.tif (shape {})'.format(psf.shape))