

import os
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

from astropy.io import fits




if __name__ == "__main__":
    spec_dir = '/home/arcadia/mysoft/gradschool/699_2/pheonix_templates/'
    cold_file = 'phoenixm00_6600.fits'
    hot_file = 'phoenixm00_8600.fits'
    
    wvln_cntl = 656.28
    dwvln = 4
    
    cold = fits.getdata(os.path.join(spec_dir, cold_file))['g40']
    hot = fits.getdata(os.path.join(spec_dir, hot_file))['g40']
    
    wvlns = fits.getdata(os.path.join(spec_dir, cold_file))['wavelength'] / 10
    
    
    mask = (wvlns > wvln_cntl - dwvln) & (wvlns < wvln_cntl + dwvln)
    wvlns = wvlns[mask]
    cold = cold[mask]
    hot = hot[mask]
    
    
    plt.plot(wvlns, cold, label='cold', color="#1680cd", linewidth=2)
    plt.plot(wvlns, hot, label='hot', color="#b44500", linewidth=2)
    plt.xlim(653, 660)
    
    plt.xlabel('λ (nm)', fontsize=14)
    plt.ylabel('Flux (erg/s/cm²/Å)', fontsize=14)
    plt.title('Hα Line Profiles', fontsize=16)
    plt.legend()
    plt.savefig('./plots/Halpha_profiles.png', dpi=300)
    
    
    
    
    
    