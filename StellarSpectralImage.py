"""
Make an spectral image of a star using the PHOENIX stellar atmosphere models. 

This script utilizes the rendered images from Stellar3DModel.py to select the 
appropriate PHOENIX spectra for a given position on the stellar disk. 
PHOENIX spectra are chosen based on the local effective temperature and surface 
gravity at each pixel, then red/blue shifted according to the local radial 
velocity.

The output of the code is a spectral image of the star (x, y, λ)
"""

import os
import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.constants import c 


# Define the parameters for the PHOENIX spectra that we have saved
SPECTRA_DIR = '/home/arcadia/mysoft/gradschool/699_2/pheonix_templates'
GRAVITATIONAL_ACCELERATIONS = np.arange(0.0, 5.1, 0.5)
TEMPERATURES = np.concatenate(
        (np.arange(5700, 7001, 100), np.arange(7000, 8801, 200))
    )
METALLICITIES = np.array([0.0])  # only solar metallicity for now




def yield_pixel_indicies(shape):
    """Generator that yields (i, j) pixel indices for a given image shape."""
    for i in range(shape[0]):
        for j in range(shape[1]):
            yield i, j



def dv_broadening_kernel(wave, dv):
    """
    Each pixel has a local radial velocity widht dv, which causes broadening 
    within a single pixel. We can approximate this as a top-hat function. 
    """
    dw = dv / c * 656.46 #np.mean(wave)  # convert velocity width to wavelength width
    dw_grid = np.diff(wave).mean()  # assume uniform wavelength grid and find grid spacing
    
    # Center the kernel on the wavelength grid and normalize
    kernel = np.zeros_like(wave)
    kernel_width = int(np.ceil(dw / dw_grid))  # number of grid points to cover the width of the kernel
    center_idx = len(wave) // 2  # center the kernel in the middle
    kernel[center_idx - kernel_width//2 : center_idx + kernel_width//2 + 1] = 1.0  # set the kernel to 1 within the width range
    
    # kernel = np.where(np.abs(wave - 656.46) <= dw/2, 1.0, 0.0)
    kernel /= np.sum(kernel)  # normalize the kernel
    return kernel





def build_spectral_image(
    render_dir="./renders",
    wave_file="./reference_maps/cm_wavelengths.fits",
    spectra_dir="/home/arcadia/mysoft/gradschool/699_2/pheonix_templates",
    wave_cntl=656.28,
    delta_wave=20.0,
    star_metallicity=0.0,
    save=True,
    save_dir=None,
    file_name = "stellar_spectral_image.fits",
):
    """
    Build a spectral image of the star using PHOENIX templates based on the
    rendered T, g and v_rad maps produced by Stellar3DModel.

    Parameters
    ----------
    render_dir : str
        Directory containing rendered_temperature.fits, rendered_gravity.fits
        and rendered_radial_velocity.fits.
    wave_file : str
        FITS file with the target wavelength grid (nm).
    spectra_dir : str
        Directory containing the PHOENIX template FITS files.
    wave_cntl, delta_wave : float
        Central wavelength and half-width (nm) of the spectral window.
    star_metallicity : float
        Metallicity of the star (currently only solar is supported).
    save : bool
        If True, write stellar_spectral_image.fits and wavelength_grid.fits.
    save_dir : str or None
        Directory in which to save the products (defaults to render_dir).

    Returns
    -------
    spectral_image : ndarray, shape (n_wave, ny, nx)
        Spectral image (flux units erg s⁻¹ cm⁻² Å⁻¹).
    target_wave : ndarray, shape (n_wave,)
        Wavelength grid (nm) corresponding to the first axis of spectral_image.
    """
    import os
    import numpy as np
    from astropy.io import fits
    from scipy.constants import c

    # PHOENIX grid definitions (kept local so the function is self-contained)
    GRAVITATIONAL_ACCELERATIONS = np.arange(0.0, 5.1, 0.5)
    # TEMPERATURES = np.concatenate(
    #     (np.arange(5700, 7001, 100), np.arange(7000, 8801, 200))
    # )
    global TEMPERATURES
    METALLICITIES = np.array([0.0])

    def yield_pixel_indices(shape):
        for i in range(shape[0]):
            for j in range(shape[1]):
                yield i, j

    def dv_broadening_kernel(wave, dv):
        dw = dv / c * 656.46
        dw_grid = np.diff(wave).mean()
        kernel = np.zeros_like(wave)
        kernel_width = int(np.ceil(dw / dw_grid))
        center_idx = len(wave) // 2
        kernel[center_idx - kernel_width // 2 : center_idx + kernel_width // 2 + 1] = 1.0
        kernel /= np.sum(kernel)
        return kernel

    if save_dir is None:
        save_dir = render_dir
    os.makedirs(save_dir, exist_ok=True)

    metallicity_idx = np.argmin(np.abs(METALLICITIES - star_metallicity))

    # Wavelength grid
    target_wave = fits.getdata(wave_file)
    wave_mask = (target_wave > wave_cntl - delta_wave) & (target_wave < wave_cntl + delta_wave)
    target_wave = target_wave[wave_mask]

    T_file = os.path.join(render_dir, "rendered_temperature.fits")
    g_file = os.path.join(render_dir, "rendered_gravity.fits")
    v_rad_file = os.path.join(render_dir, "rendered_radial_velocity.fits")

    hdr = fits.getheader(T_file)
    try:
        PXSCALE = hdr["PXSCALE"]
        PXUNIT  = hdr["PXUNIT"]
    except KeyError:
        # fall-back (should never be needed once the writer above is used)
        PXSCALE = 1.0
        PXUNIT  = "m/pixel"
        print("WARNING: PXSCALE missing – using dummy value")

    T_map = fits.getdata(T_file)
    g_map = fits.getdata(g_file)
    v_rad_map = fits.getdata(v_rad_file)

    # Median velocity gradient (for intra-pixel broadening)
    v_rad_grad = np.gradient(v_rad_map)
    dv_rad = np.sqrt(v_rad_grad[0] ** 2 + v_rad_grad[1] ** 2)
    dv_rad = np.median(dv_rad[~np.isnan(dv_rad)])

    spectral_image = np.zeros((target_wave.size, *T_map.shape))
    print("constructing spectral image...")
    for i, j in yield_pixel_indices(T_map.shape):
        print(f"Processing pixel ({i}, {j})", end="\r", flush=True)
        local_T = T_map[i, j]
        local_g = g_map[i, j]
        local_v_rad = v_rad_map[i, j]

        if np.any(np.isnan([local_T, local_g, local_v_rad])):
            continue

        temp_idx = np.argmin(np.abs(TEMPERATURES - local_T))
        grav_idx = np.argmin(np.abs(GRAVITATIONAL_ACCELERATIONS - local_g))

        spec_file = (
            f"phoenixm{int(METALLICITIES[metallicity_idx] * 10):02d}_"
            f"{TEMPERATURES[temp_idx]:0d}.fits"
        )
        spec_filepath = os.path.join(spectra_dir, spec_file)

        g_key = f"g{GRAVITATIONAL_ACCELERATIONS[grav_idx] * 10:02.0f}"
        spec = fits.getdata(spec_filepath)[g_key]
        spec_wave = fits.getdata(spec_filepath)["WAVELENGTH"] / 10.0

        # Crop for speed
        wave_mask_spec = (spec_wave > wave_cntl - delta_wave - 10) & (
            spec_wave < wave_cntl + delta_wave + 10
        )
        spec_wave = spec_wave[wave_mask_spec]
        spec = spec[wave_mask_spec]

        # Broaden
        v_broad_kernel = dv_broadening_kernel(spec_wave, dv_rad * 1e3)
        spec = np.convolve(spec, v_broad_kernel, mode="same")
        assert not np.all(np.isnan(spec)), (
            "Velocity broadening kernel is too narrow. "
            "Extrapolate PHOENIX spectra to a finer wavelength grid or "
            "reduce spatial resolution of the model."
        )

        # Doppler shift
        spec_wave_shifted = spec_wave + spec_wave * (local_v_rad * 1e3 / c)

        # Interpolate onto target grid
        if target_wave[0] > target_wave[-1]:
            interp_spec = np.interp(target_wave[::-1], spec_wave_shifted, spec)[::-1]
        else:
            interp_spec = np.interp(target_wave, spec_wave_shifted, spec)

        spectral_image[:, i, j] = interp_spec

    if save:
        hdu = fits.PrimaryHDU(spectral_image)
        hdu.header["BUNIT"] = "erg/s/cm^2/Angstrom"
        hdu.header["PXUNIT"] = PXUNIT
        hdu.header["PXSCALE"] = PXSCALE
        hdu.header["COMMENT"] = (
            "Spectral image of star generated from PHOENIX spectra "
            "based on local T, g, and v_rad maps"
        )
        hdu.writeto(
            os.path.join(save_dir, file_name), overwrite=True
        )

        wave_hdu = fits.PrimaryHDU(target_wave)
        wave_hdu.header["BUNIT"] = "nm"
        wave_hdu.header["COMMENT"] = "Wavelength grid for stellar spectral image"
        wave_hdu.writeto(
            os.path.join(save_dir, "wavelength_grid.fits"), overwrite=True
        )

    return spectral_image, target_wave




if __name__ == "__main__":
    save_dir = './renders'
    # metallicity of our star
    star_metallicity = 0.0 
    # Wavelength grid to interpolate the spectra onto
    wave_file = './reference_maps/cm_wavelengths.fits'
    # Bounds of wavlength grid (nm)
    wave_cntl = 656.28
    delta_wave = 20
    # Images defining local stellar surface properties (from Stellar3DModel.py)
    T_file = './renders/rendered_temperature.fits'
    g_file = './renders/rendered_gravity.fits'
    v_rad_file = './renders/rendered_radial_velocity.fits'
    dim_fac_file = './renders/limb_darken_dimming_factor.fits'
    
    
    os.makedirs(save_dir, exist_ok=True)
    # Find the closest PHOENIX grid point for our star's metallicity
    # (for now we only have solar metallicity, so it doesn't really matter)
    metallicity_idx = np.argmin(np.abs(METALLICITIES - star_metallicity))
    
    
    # Open the wavelength grid
    target_wave = fits.getdata(wave_file)
    wave_mask = (target_wave > wave_cntl - delta_wave) & (target_wave < wave_cntl + delta_wave)
    target_wave = target_wave[wave_mask]
    PXUNIT = fits.getheader(T_file)['PXUNIT']
    PXSCALE = fits.getheader(T_file)['PXSCALE']
    
    # open the rendered images from Stellar3DModel.py
    T_map = fits.getdata(T_file)
    g_map = fits.getdata(g_file)
    v_rad_map = fits.getdata(v_rad_file)
    dim_fac_map = fits.getdata(dim_fac_file)
    
    # Compute the width of each velocity bin
    v_rad_grad = np.gradient(v_rad_map)
    dv_rad = np.sqrt(v_rad_grad[0]**2 + v_rad_grad[1]**2)
    dv_rad = np.median(dv_rad[~np.isnan(dv_rad)])  # use median to avoid outliers
    
    
    
    spectral_image = np.zeros((target_wave.size, *T_map.shape))
    
    for i, j in yield_pixel_indicies(T_map.shape):
        print(f"working on pixel ({i}, {j})", end="\r", flush=True)
        local_T = T_map[i, j]
        local_g = g_map[i, j]
        local_v_rad = v_rad_map[i, j]
        local_dim_fac = dim_fac_map[i, j]
        
        if np.any(np.isnan([local_T, local_g, local_v_rad])):
            continue  # skip pixels with missing data
        
        # Find the closest PHOENIX grid point for this pixel's T and g
        temp_idx = np.argmin(np.abs(TEMPERATURES - local_T))
        grav_idx = np.argmin(np.abs(GRAVITATIONAL_ACCELERATIONS - local_g))
        
        # Construct the filename for the corresponding PHOENIX spectrum
        spec_file = f'phoenixm{int(METALLICITIES[metallicity_idx] * 10):02d}_{TEMPERATURES[temp_idx]:0d}.fits'
        spec_filepath = os.path.join(SPECTRA_DIR, spec_file)
        
        g_key = f'g{GRAVITATIONAL_ACCELERATIONS[grav_idx]*10:02.0f}'
        spec = fits.getdata(spec_filepath)[g_key]
        spec_wave = fits.getdata(spec_filepath)['WAVELENGTH'] / 10 
        
        # Crop within near the wavelength range of interest to speed up interpolation
        wave_mask_spec = (spec_wave > wave_cntl - delta_wave-10) & (spec_wave < wave_cntl + delta_wave+10)
        spec_wave = spec_wave[wave_mask_spec]
        spec = spec[wave_mask_spec]
        
        # Broaden the spectrum according to the local velocity gradient (dv_rad)
        v_broad_kernel = dv_broadening_kernel(spec_wave, dv_rad*1e3)
        spec = np.convolve(spec, v_broad_kernel, mode='same')
        # If the spectrum is all zeros after this step, it is because the 
        # dv_rad kernel is too narrow for the wavelength grid. In that case, 
        # extrapolate the spectrum to a finer wavelength grid and re-convolve.
        assert not np.all(np.isnan(spec)); "Velocity broadening Kernel is too narrow. You may need to extrapolate the PHOENIX spectra to a finer wavelength grid before convolution, or reduce spatial resolution of model"
        
        # Red/blue shift the spectrum according to the local radial velocity
        spec_wave_shifted = spec_wave + spec_wave * (local_v_rad*1e3 / c)
        
        # Interpolate the shifted spectrum onto the target wavelength grid
        # if wavelength grid is in reverse order, flip it and the spectrum for interpolation
        if target_wave[0] > target_wave[-1]:  
            interp_spec = np.interp(target_wave[::-1], spec_wave_shifted, spec)[::-1]
        else:
            interp_spec = np.interp(target_wave, spec_wave_shifted, spec)
        
        # Insert the spectrum into this pixel, and apply limb darkening
        spectral_image[:, i, j] = interp_spec * local_dim_fac
        
        # plt.plot(spec_wave_shifted, spec, label='Shifted PHOENIX spectrum')
        # plt.plot(spec_wave, spec, label='Original PHOENIX spectrum')
        # plt.plot(target_wave, interp_spec, label='Interpolated and broadened spectrum')
        # plt.plot()
        # plt.xlim(wave_cntl - delta_wave, wave_cntl + delta_wave)
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
        # plt.show()
        # break  # just do one pixel for testing    
    
    
    # # Structure the spectral image as a 3D FITS cube with axes (λ, x, y)
    # spectral_image = np.transpose(spectral_image, (2, 0, 1))
    
    # Save the spectral image as a FITS file
    hdu = fits.PrimaryHDU(spectral_image)
    hdu.header['BUNIT'] = 'erg/s/cm^2/Angstrom'
    hdu.header['PXUNIT'] = PXUNIT
    hdu.header['PXSCALE'] = PXSCALE
    hdu.header['COMMENT'] = 'Spectral image of star generated from PHOENIX spectra based on local T, g, and v_rad maps'
    savepath = os.path.join(save_dir, 'stellar_spectral_image.fits')
    hdu.writeto(savepath, overwrite=True)
    
    # Save the wavelength grid as a separate FITS file
    wave_hdu = fits.PrimaryHDU(target_wave)
    wave_hdu.header['BUNIT'] = 'nm'
    wave_hdu.header['COMMENT'] = 'Wavelength grid for stellar spectral image'
    wave_savepath = os.path.join(save_dir, 'wavelength_grid.fits')
    wave_hdu.writeto(wave_savepath, overwrite=True)
    
    
    # %%
    
    # plot the full disk-integrated spectrum for sanity check
    # disk_integrated_spectrum = np.sum(spectral_image, axis=(1,2))
    # plt.plot(target_wave, disk_integrated_spectrum)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Flux (erg/s/cm^2/Angstrom)')
    # plt.title('Disk-integrated spectrum of the star')
    # plt.xlim(wave_cntl - delta_wave, wave_cntl + delta_wave)
    # plt.savefig('./plots/disk_integrated_spectrum_simulated.png', dpi=300)