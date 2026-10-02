"""
Deshift the maps observed onsky to get the "theoretical" response of the 
lantern + noise. 

We then will convolve these maps with the rendered stellar spectral image to 
get the observed coupling maps.

"""
import os
import sys 
import glob
from astropy.io import fits
import numpy as np

sys.path.append('/home/arcadia/mysoft/gradschool/699_2/reduction_tools/my_method/')
from coupling_map import Spectral_Coupling_Map
from CouplingMapModel import CouplingMapModel, measure_shifts

MODEL_BOUNDARY = (652, 660)  # nm, wavelength range to use for measuring shifts




def deshift_map(coupling_map, wvlns, shifts, shift_wavs):
    """
    Deshift the observed coupling map by the measured shift to match the 
    reference map. 
    """
    deshifted_map = coupling_map.new_with_data(coupling_map.data)
        
    for i, wav in enumerate(shift_wavs):
        wvln_idx = np.argmin(np.abs(wvlns - wav))
        dx = shifts[i][0]
        dy = shifts[i][1]
        print(f"Deshifting wavelength {wav:.2f} nm (index {wvln_idx}) by dx={dx:.3f}, dy={dy:.3f}")
        deshifted_map[wvln_idx] = coupling_map.translate_slice(wvln_idx, -dx, -dy)
    
    return deshifted_map



if __name__ == "__main__":
    cm_dir = './onsky_maps/'
    rm_dir = './reference_maps/'
    wave_file = './reference_maps/cm_wavelengths.fits'
    
    wvlns = fits.getdata(wave_file)
    cm_files = sorted(glob.glob(os.path.join(cm_dir, 'coupling_map_*.fits')))
    rm_files = sorted(glob.glob(os.path.join(rm_dir, 'reference_map_*.fits')))
    
    
    cms = [Spectral_Coupling_Map.readfrom(file) for file in cm_files]
    rms = [Spectral_Coupling_Map.readfrom(file) for file in rm_files]
    
    
    results, wavs = measure_shifts(
        cms,
        rms,
        wvlns, 
        MODEL_BOUNDARY
    )
    
    
    # De-shift the observed maps to match the reference maps
    for i, cm in enumerate(cms):
        cm_deshifted = deshift_map(cm, wvlns, results, wavs)
        savefile = os.path.basename(cm_files[i]).replace('coupling_map', 'deshifted_map')
        cm_deshifted.writeto(os.path.join('./deshifted_maps/', savefile))