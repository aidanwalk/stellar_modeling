"""
Compute the theoretical centroid shifts by convolving the rendered stellar 
spectral image with the theoretical response maps

"""

import os
import sys
import cv2
import glob
import numpy as np
from astropy.io import fits
# import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.signal import fftconvolve


sys.path.append('/home/arcadia/mysoft/gradschool/699_2/reduction_tools/my_method/')
from coupling_map import Spectral_Coupling_Map





def zoom_reference_map(map, zoom_factor, order=3):
    """Zoom the reference maps to match the angular scale of the spectral image"""
    
    # Since the reference maps contain nan, we first need to mask these values, 
    # then zoom both the data and the mask, and then re-apply the mask after zooming
        
    new_map = np.zeros((map.shape[0], round(map.shape[1]*zoom_factor), round(map.shape[2]*zoom_factor)))
    for wvln_idx in range(map.shape[0]):
        data_slice = map[wvln_idx, :, :]
        mask = np.isnan(data_slice)
        
        # interpolate the nan in data slice before zooming to avoid creating large blocks of nans after zooming
        x = np.arange(data_slice.shape[1])
        y = np.arange(data_slice.shape[0])
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx[~mask], yy[~mask]))
        values = data_slice[~mask]
        grid_x, grid_y = np.meshgrid(np.arange(data_slice.shape[1]), np.arange(data_slice.shape[0]))
        data_slice = griddata(points, values, (grid_x, grid_y), method='nearest')
        
        # zoomed_data_slice = cv2.resize(data_slice, (new_map.shape[2], new_map.shape[1]), interpolation=cv2.INTER_CUBIC)
        zoomed_data_slice = zoom(data_slice, zoom_factor, order=order, mode='reflect', grid_mode=True)
        # zoomed_mask_slice = zoom(mask.astype(float), zoom_factor, order=0) > 0.5  # zoom the mask and threshold it
        # zoomed_data_slice[zoomed_mask_slice] = np.nan  # re-apply the mask after zooming
        new_map[wvln_idx, :, :] = zoomed_data_slice
        
    new_map = map.new_with_data(new_map)
    return new_map




def convolve_map_with_image(reference_map, spectral_image):
    observed_map = reference_map.new_with_data(np.zeros(reference_map.shape))
    
    # normalize the star image to preserve flux
    star_image = spectral_image / np.nansum(spectral_image, axis=(1,2))[:, np.newaxis, np.newaxis]
    
    observed_map = fftconvolve(reference_map, star_image, mode='same', axes=(1,2))
            
    return observed_map








def convolve_all_maps(
    spectral_image_file="./renders/stellar_spectral_image.fits",
    wavelength_grid_file="./renders/wavelength_grid.fits",
    maps_dir="./deshifted_maps",
    maps_wavefile="./reference_maps/cm_wavelengths.fits",
    savedir="./observed_maps",
    wvln_cntl=656.28,
    delta_wvln=5.0,
    altair_pllx=194.95e-3,          # arcsec
    plate_scale=5.9e-3,             # arcsec / pixel (VAMPIRES)
    verbose=False,
):
    """
    Convolve a stellar spectral image with a set of deshifted reference
    coupling maps to produce simulated observed coupling maps.

    Parameters
    ----------
    spectral_image_file : str
        Path to the 3-D FITS cube (λ, y, x) produced by build_spectral_image.
    wavelength_grid_file : str
        Path to the corresponding 1-D wavelength grid FITS file (nm).
    maps_dir : str
        Directory containing deshifted_map_*.fits files.
    maps_wavefile : str
        FITS file with the wavelength grid of the reference/deshifted maps.
    savedir : str
        Directory in which the resulting coupling_map_*.fits files are written.
    wvln_cntl, delta_wvln : float
        Central wavelength and half-width (nm) used to crop both the spectral
        image and the reference maps.
    altair_pllx : float
        Parallax of the star (arcsec). Used to convert the physical pixel scale
        of the spectral image into an angular scale.
    plate_scale : float
        Angular plate scale of the focal-plane camera (arcsec / pixel).
    verbose : bool
        Print progress messages.

    Returns
    -------
    list of str
        Paths of the written coupling_map_*.fits files.
    """

    # ------------------------------------------------------------------
    # Helper functions (kept inside so the callable is self-contained)
    # ------------------------------------------------------------------
    def zoom_reference_map(map_, zoom_factor, order=3):
        """Zoom a Spectral_Coupling_Map to a new angular scale, handling NaNs."""
        new_data = np.zeros(
            (
                map_.shape[0],
                round(map_.shape[1] * zoom_factor),
                round(map_.shape[2] * zoom_factor),
            )
        )
        for wvln_idx in range(map_.shape[0]):
            data_slice = map_[wvln_idx, :, :]
            mask = np.isnan(data_slice)

            # Fill NaNs before zooming
            x = np.arange(data_slice.shape[1])
            y = np.arange(data_slice.shape[0])
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack((xx[~mask], yy[~mask]))
            values = data_slice[~mask]
            grid_x, grid_y = np.meshgrid(
                np.arange(data_slice.shape[1]), np.arange(data_slice.shape[0])
            )
            data_slice = griddata(points, values, (grid_x, grid_y), method="nearest")

            zoomed = zoom(
                data_slice, zoom_factor, order=order, mode="reflect", grid_mode=True
            )
            new_data[wvln_idx, :, :] = zoomed

        return map_.new_with_data(new_data)

    def convolve_map_with_image(reference_map, spectral_image):
        """Convolve a (zoomed) reference map with a normalised spectral image."""
        star_image = spectral_image / np.nansum(
            spectral_image, axis=(1, 2)
        )[:, np.newaxis, np.newaxis]
        return fftconvolve(reference_map, star_image, mode="same", axes=(1, 2))

    # ------------------------------------------------------------------
    # Main logic
    # ------------------------------------------------------------------
    os.makedirs(savedir, exist_ok=True)

    spectral_image = fits.getdata(spectral_image_file)
    spectral_image_scale = fits.getheader(spectral_image_file)["PXSCALE"]  # m/pixel
    sim_wav = fits.getdata(wavelength_grid_file)

    # Angular scale of the simulated image
    d = 1.0 / altair_pllx * 3.086e16  # metres
    spectral_image_angular_scale = spectral_image_scale / d * 206265  # arcsec/pixel
    if verbose:
        print(f"{spectral_image_angular_scale * 1e6:.2f} microarcsec per pixel")

    # Crop spectral image to the requested wavelength window
    wave_mask = (sim_wav > wvln_cntl - delta_wvln) & (sim_wav < wvln_cntl + delta_wvln)
    sim_wave = sim_wav[wave_mask]
    spectral_image = spectral_image[wave_mask, :, :]

    # Reference / deshifted maps
    reference_files = sorted(glob.glob(os.path.join(maps_dir, "deshifted_map_*.fits")))
    ref_wav = fits.getdata(maps_wavefile)
    wave_mask = (ref_wav > wvln_cntl - delta_wvln) & (ref_wav < wvln_cntl + delta_wvln)
    ref_wav_masked = ref_wav[wave_mask]
    assert np.allclose(sim_wave, ref_wav_masked), (
        "Wavelength grids of spectral image and reference maps do not match."
    )

    written_files = []
    for file in reference_files:
        if verbose:
            print(f"Opening reference map: {file}")
        reference_map = Spectral_Coupling_Map.readfrom(file)
        reference_map_angular_scale = plate_scale / reference_map.bins_per_pixel  # arcsec/pixel

        # Crop to the wavelength window
        reference_map_masked = reference_map.new_with_data(
            reference_map[wave_mask, :, :]
        )

        # Zoom so that the angular scales match
        zoom_factor = reference_map_angular_scale / spectral_image_angular_scale
        if verbose:
            print(f"Zoom factor: {zoom_factor:.2f}")
            print("Zooming reference map …")
        zoomed_map = zoom_reference_map(reference_map_masked, zoom_factor)

        if verbose:
            print("Convolving zoomed map with spectral image …")
        convolved_map = convolve_map_with_image(zoomed_map, spectral_image)

        # Resize back to the original (masked) spatial size
        if verbose:
            print("Resizing convolved map back to original scale …")
        observed_map = reference_map_masked.new_with_data(
            np.zeros(reference_map_masked.data.shape)
        )
        for wvln_idx in range(reference_map_masked.shape[0]):
            observed_map_slice = cv2.resize(
                convolved_map[wvln_idx],
                (reference_map_masked.shape[2], reference_map_masked.shape[1]),
                interpolation=cv2.INTER_CUBIC,
            )
            observed_map[wvln_idx, :, :] = observed_map_slice
        observed_map[np.isnan(reference_map_masked)] = np.nan

        # Embed into the full (un-masked) wavelength grid
        observed_map_full = reference_map.new_with_data(
            np.zeros(reference_map.shape)
        )
        observed_map_full[wave_mask, :, :] = observed_map.data
        observed_map_full[np.isnan(reference_map)] = np.nan

        # Write
        savefile = os.path.basename(file).replace("deshifted_map", "coupling_map")
        outpath = os.path.join(savedir, savefile)
        if verbose:
            print(f"Saving observed map to {outpath}")
        observed_map_full.writeto(outpath)
        written_files.append(outpath)

    return written_files


            
# %%

if __name__ == "__main__":
    savedir = './observed_maps'
    wvln_cntl = 656.28
    delta_wvln = 5

    ALTAIR_PLLX = 194.95e-3 # arcseconds
    PLATE_SCALE = 5.9e-3 # plate scale of focal plane camera (vampires)


    # First, open the star spectral image and wavelength grid
    stellar_spectral_image_file = './renders/stellar_spectral_image.fits'
    wavelength_grid_file = './renders/wavelength_grid.fits'

    maps_dir = './reference_maps'
    maps_wavefile = './reference_maps/cm_wavelengths.fits'

    
    os.makedirs(savedir, exist_ok=True)
    
    
    spectral_image = fits.getdata(stellar_spectral_image_file)
    spectral_image_scale = fits.getheader(stellar_spectral_image_file)['PXSCALE']  # m/pixel
    sim_wav = fits.getdata(wavelength_grid_file)

    # Find the angular scale of the simulated image using distance to object
    d = 1 / (ALTAIR_PLLX) * 3.086e16  
    spectral_image_angular_scale = spectral_image_scale / d  * 206265 # arcseconds per pixel
    print(spectral_image_angular_scale*1e6, "microarcseconds per pixel")
    
    # Crop the spectral image to only include wavelengths within a certain range around the central wavelength
    wave_mask = (sim_wav > wvln_cntl - delta_wvln) & (sim_wav < wvln_cntl + delta_wvln)
    sim_wave = sim_wav[wave_mask]
    spectral_image = spectral_image[wave_mask, :, :]
    
    
    # Find reference map file names
    # reference_files = sorted(glob.glob(os.path.join(maps_dir, 'deshifted_map_*.fits')))
    reference_files = sorted(glob.glob(os.path.join(maps_dir, 'reference_map_*.fits')))
    ref_wav = fits.getdata(maps_wavefile)
    wave_mask = (ref_wav > wvln_cntl - delta_wvln) & (ref_wav < wvln_cntl + delta_wvln)
    ref_wav_masked = ref_wav[wave_mask]
    assert np.allclose(sim_wave, ref_wav_masked), "Wavelength grids of spectral image and reference maps do not match."
    
    
    for file in reference_files:
        print(f"Opening reference map: {file}")
        reference_map = Spectral_Coupling_Map.readfrom(file)
        reference_map_angular_scale = PLATE_SCALE / reference_map.bins_per_pixel  # arcseconds per pixel

        # Crop the reference map to only include the wavelengths we want
        reference_map_masked = reference_map.new_with_data(reference_map[wave_mask, :, :])
        
        # Zoom the refernce maps to match the angular scale of the spectral image
        zoom_factor = reference_map_angular_scale / spectral_image_angular_scale
        print(f"Zoom factor: {zoom_factor:.2f}")
        print("Zooming Reference map")
        zoomed_map = zoom_reference_map(reference_map_masked, zoom_factor)
        
        
        print("Convolving zoomed map with spectral image")
        convolved_map = convolve_map_with_image(zoomed_map, spectral_image)

        print("Zooming convolved map back to original scale")
        observed_map = reference_map_masked.new_with_data(np.zeros(reference_map_masked.data.shape))
        for wvln_idx in range(reference_map_masked.shape[0]):
            observed_map_slice = cv2.resize(
                convolved_map[wvln_idx],
                (reference_map_masked.shape[2], reference_map_masked.shape[1]),
                interpolation=cv2.INTER_CUBIC
            )
            observed_map[wvln_idx,:,:] = observed_map_slice
        # Reapply nan pixels to the observed map
        observed_map[np.isnan(reference_map_masked)] = np.nan

        
        # We want observed map to be on the same wavelength grid as the 
        # reference map, so we can directly compare them and measure centroid shifts.
        observed_map_full = reference_map.new_with_data(np.zeros(reference_map.shape))
        observed_map_full[wave_mask, :, :] = observed_map.data
        observed_map_full[np.isnan(reference_map)] = np.nan  # reapply nans to the full observed map
        
        # Save the observed map
        savefile = os.path.basename(file).replace('deshifted_map', 'coupling_map')
        print(f"Saving observed map to {os.path.join(savedir, savefile)}")
        observed_map_full.writeto(os.path.join(savedir, savefile))
    
    