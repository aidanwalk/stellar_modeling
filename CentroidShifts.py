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
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from scipy.signal import fftconvolve


sys.path.append('/home/arcadia/mysoft/gradschool/699_2/reduction_tools/my_method/')
from coupling_map import Spectral_Coupling_Map
from CouplingMapModel import measure_shifts, CouplingMapModel




def measure_centroid_shifts(
    cm_dir="./observed_maps/",
    ref_dir="./reference_maps/",
    ref_wavefile="./reference_maps/cm_wavelengths.fits",
    model_boundary=(652.0, 660.0),
    plate_scale=5.9,               # mas / pixel
    savefile=None,
    verbose=False,
):
    """
    Measure photocentre (centroid) shifts between a set of simulated/observed
    coupling maps and the corresponding theoretical reference maps.

    Parameters
    ----------
    cm_dir : str
        Directory containing coupling_map_*.fits files
        (the maps produced by convolve_all_maps, or real on-sky maps).
    ref_dir : str
        Directory containing reference_map_*.fits files.
    ref_wavefile : str
        FITS file with the common wavelength grid (nm).
    model_boundary : tuple of float
        (λ_min, λ_max) in nm – wavelength window passed to measure_shifts.
    plate_scale : float
        Conversion factor from pixels to milliarcseconds.
    savefile : str or None
        If given, the result (wavelength + shifts) is written as a .npy file.
    verbose : bool
        Print progress messages.

    Returns
    -------
    wavs : ndarray, shape (N,)
        Wavelengths (nm) at which the shifts were evaluated.
    results : ndarray, shape (N, n_components)
        Mean-subtracted centroid shifts in milliarcseconds.
        (Typically two columns: dx, dy or the major-axis projection.)
    """
    import os
    import sys
    import glob
    import numpy as np
    from astropy.io import fits

    # Local imports of the coupling-map classes
    sys.path.append("/home/arcadia/mysoft/gradschool/699_2/reduction_tools/my_method/")
    from coupling_map import Spectral_Coupling_Map
    from CouplingMapModel import measure_shifts

    cm_files = sorted(glob.glob(os.path.join(cm_dir, "coupling_map_*.fits")))
    ref_files = sorted(glob.glob(os.path.join(ref_dir, "reference_map_*.fits")))

    if len(cm_files) == 0:
        raise FileNotFoundError(f"No coupling_map_*.fits found in {cm_dir}")
    if len(ref_files) == 0:
        raise FileNotFoundError(f"No reference_map_*.fits found in {ref_dir}")
    if len(cm_files) != len(ref_files):
        raise ValueError(
            f"Number of coupling maps ({len(cm_files)}) does not match "
            f"number of reference maps ({len(ref_files)})"
        )

    wvlns = fits.getdata(ref_wavefile)

    coupling_maps = []
    reference_maps = []
    for cm_file, ref_file in zip(cm_files, ref_files):
        if verbose:
            print(f"Loading {os.path.basename(cm_file)}  ↔  {os.path.basename(ref_file)}")
        cm = Spectral_Coupling_Map.readfrom(cm_file)
        ref = Spectral_Coupling_Map.readfrom(ref_file)
        coupling_maps.append(cm)
        reference_maps.append(ref)

    # Core measurement (imported from CouplingMapModel)
    results, wavs = measure_shifts(
        coupling_maps,
        reference_maps,
        wvlns,
        model_boundary,
    )

    results = np.asarray(results) * plate_scale          # → mas
    results -= np.mean(results, axis=0)                  # zero the mean shift

    if savefile is not None:
        os.makedirs(os.path.dirname(savefile) or ".", exist_ok=True)
        np.save(savefile, np.hstack((wavs[:, None], results)))
        if verbose:
            print(f"Saved centroid shifts to {savefile}")

    return wavs, results



if __name__ == "__main__":
    PLATE_SCALE = 5.9  # mas/pixel
    MODEL_BOUNDARY = (652, 660)
    SAVEFILE = './observed_maps/centroid_shifts.npy'
    cm_dir = './observed_maps/'
    ref_dir = './reference_maps/'
    ref_wavefile = './reference_maps/cm_wavelengths.fits'
    
    
    cm_files = sorted(glob.glob(os.path.join(cm_dir, 'coupling_map_*.fits')))
    wvlns = fits.getdata(ref_wavefile)
    
    ref_files = sorted(glob.glob(os.path.join(ref_dir, 'reference_map_*.fits')))
    
    
    coupling_maps = []
    reference_maps = []
    for cm_file, ref_file in zip(cm_files, ref_files):
        cm = Spectral_Coupling_Map.readfrom(cm_file)
        ref = Spectral_Coupling_Map.readfrom(ref_file)
        coupling_maps.append(cm)
        reference_maps.append(ref)
        
    
    # Compute the optimal shift among all port's measured vs theoretical maps
    # as a function of wavelength 
    results, wavs = measure_shifts(
        coupling_maps,
        reference_maps,
        wvlns, 
        MODEL_BOUNDARY
    )


    results = np.array(results) * PLATE_SCALE # convert to mas
    # errors = np.array(errors) * PLATE_SCALE
    results -= np.mean(results, axis=0)  # zero the mean shift
    
    np.save(SAVEFILE, np.hstack((wavs[:, None], results)))
    