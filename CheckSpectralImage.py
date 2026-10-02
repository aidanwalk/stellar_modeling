


import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt




def mask_spectrum(wave, spec, wave_cntl, delta_wave):
    """Mask the spectrum to only include wavelengths within a certain range."""
    mask = (wave > wave_cntl - delta_wave) & (wave < wave_cntl + delta_wave)
    return wave[mask], spec[mask]




wave_cntl = 656.28
delta_wave = 20



observed_spec_file = '/home/arcadia/mysoft/gradschool/699_2/reduction_tools/my_method/line_fitting/collapsed_spectrum.npy'
obs_wave, obs_spec = np.load(observed_spec_file)


model_image_file = './renders/stellar_spectral_image.fits'
model_image_wave_file = './renders/wavelength_grid.fits'
model_image = fits.getdata(model_image_file)
model_wave = fits.getdata(model_image_wave_file)

model_spec = np.sum(model_image, axis=(1,2))


obs_wave, obs_spec = mask_spectrum(obs_wave, obs_spec, wave_cntl, delta_wave)
model_wave, model_spec = mask_spectrum(model_wave, model_spec, wave_cntl, delta_wave)


# Fit a line to the continuum and normalize both spectra by the continuum level at the control wavelength
dw = 5
continuum_mask_obs = (obs_wave < wave_cntl - dw) | (obs_wave > wave_cntl + dw)
continuum_mask_model = (model_wave < wave_cntl - dw) | (model_wave > wave_cntl + dw)

# Fit a line to the continuum regions
obs_continuum_fit = np.polyfit(obs_wave[continuum_mask_obs], obs_spec[continuum_mask_obs], deg=4)
model_continuum_fit = np.polyfit(model_wave[continuum_mask_model], model_spec[continuum_mask_model], deg=4)

obs_continuum = np.polyval(obs_continuum_fit, obs_wave)
model_continuum = np.polyval(model_continuum_fit, model_wave)
obs_spec = obs_spec / obs_continuum
model_spec = model_spec / model_continuum





plt.figure(figsize=(6, 4), tight_layout=True)
plt.title('Disk-integrated spectrum of Altair')

plt.plot(obs_wave+0.3, obs_spec, label='Observed spectrum', color='k', alpha=0.5)
plt.plot(model_wave, model_spec, label='Modeled spectrum', color='#008cff', linewidth=2)
plt.legend()

plt.xlabel('Wavelength (nm)')
plt.ylabel('Flux (norm)')

plt.savefig('./plots/disk_integrated_spectrum.png', dpi=300)