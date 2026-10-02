#!/usr/bin/env python
"""
Iterative fit of omega_frac, inclination and R_p to an observed
spectro-astrometric centroid-shift signal.

The forward model is the full simulation chain:
  Stellar3DModel → StellarSpectralImage → ConvolveMaps → CentroidShifts
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from scipy.optimize import differential_evolution, minimize
from astropy.constants import M_sun, R_sun
from astropy.io import fits
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths – adjust to your machine
# ---------------------------------------------------------------------------
OBSERVED_FILE = (
    "/home/arcadia/mysoft/gradschool/dissertation/papers/"
    "humu_2026/figures/centroid_shifts_major_axis/centroid_shifts.npy"
)  # or whatever file actually contains (wav, dx[, dy])

WORK_ROOT = "./fit_workdir"          # temporary directories live here
RESULTS_DIR = "./fit_results"
os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed stellar parameters (Altair)
M_STAR = 1.791 * M_sun.value
T_EFF  = 7550.0
BETA   = 0.25
PA     = -62.7                       # position angle (deg) – usually well known

# Wavelength window used by the spectro-astrometry pipeline
MODEL_BOUNDARY = (652.0, 660.0)      # nm
PLATE_SCALE    = 5.9                 # mas / pixel  (VAMPIRES)

# ---------------------------------------------------------------------------
# Forward-model wrapper
# ---------------------------------------------------------------------------
def run_forward_model(omega_frac, inclination, R_p,
                      workdir=None, clean=True):
    """
    Execute the full simulation chain for one parameter set and return
    the predicted centroid shifts (same format as CentroidShifts.py).

    Returns
    -------
    wavs : (N,) array
    shifts : (N, n_ports_or_2) array   [mas, mean-subtracted]
    """
    
    # global _call_count
    # _call_count = getattr(run_forward_model, "_call_count", 0) + 1
    # run_forward_model._call_count = _call_count
    # print(f"[forward { _call_count:3d}] ω={omega_frac:.4f}  i={inclination:.2f}  Rp={R_p/R_sun.value:.4f}")
    
    
    if workdir is None:
        workdir = tempfile.mkdtemp(dir=WORK_ROOT)
    else:
        os.makedirs(workdir, exist_ok=True)

    # --- 1. Render T, g, vrad maps ------------------------------------------
    #   (import locally so the temporary directory can be the CWD)
    sys.path.insert(0, os.path.dirname(__file__))
    from Stellar3DModel import Stellar3DModel

    model = Stellar3DModel(
        M_STAR, R_p, T_EFF,
        beta=BETA, omega_frac=omega_frac,
        N_grid=2**5,                 # lower resolution for speed during fit
        render_resolution=2**5,
        overfill=1.2
    )

    T_vol   = model.build_temperature_volume()
    g_vol   = model.build_gravity_volume()
    vrad_vol = model.build_radial_velocity_volume(inclination, PA)

    image_T   = model.render(T_vol,   inclination, PA, mode="surface", project_vcam=True, vcam_derot_ang=101.34)
    image_g   = model.render(g_vol,   inclination, PA, mode="surface", project_vcam=True, vcam_derot_ang=101.34)
    image_vrad = model.render(vrad_vol, inclination, PA,
                              mode="surface", threshold=1e-3, project_vcam=True, vcam_derot_ang=101.34)
    image_vrad /= 1e3                # km/s
    image_g = np.log10(image_g * 1e2)

    pixel_scale = model.get_pixel_scale(inclination, PA)

    render_dir = os.path.join(workdir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    def _write_rendered(name, data, bunit):
        """Write a rendered map with the required header keywords."""
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header["BUNIT"]   = bunit
        hdu.header["PXSCALE"] = float(pixel_scale)   # critical!
        hdu.header["PXUNIT"]  = "m/pixel"
        hdu.header["COMMENT"] = "Rendered by Stellar3DModel for spectro-astrometry fit"
        path = os.path.join(render_dir, name)
        hdu.writeto(path, overwrite=True)
        return path
    
    _write_rendered("rendered_temperature.fits",     image_T,    "K")
    _write_rendered("rendered_gravity.fits",         image_g,    "log10(cm/s^2)")
    _write_rendered("rendered_radial_velocity.fits", image_vrad, "km/s")

    # --- 2. Build spectral image --------------------------------------------
    #   (StellarSpectralImage.py expects the three FITS files above)
    #   For speed one may cache PHOENIX spectra or pre-interpolate them.
    from StellarSpectralImage import build_spectral_image   # see note below
    spectral_image, sim_wav = build_spectral_image(
        render_dir,
        wave_cntl=656.28, delta_wave=20.0,
        spectra_dir="/home/arcadia/mysoft/gradschool/699_2/pheonix_templates", 
        save=True, save_dir=render_dir,
    )
    # fits.writeto(os.path.join(render_dir, "stellar_spectral_image.fits"),
    #              spectral_image, overwrite=True)
    # fits.writeto(os.path.join(render_dir, "wavelength_grid.fits"),
    #              sim_wav, overwrite=True)

    # --- 3. Convolve with deshifted reference maps --------------------------
    #   (ConvolveMaps.py – we assume the deshifted_maps/ directory already
    #    exists from a previous run of DeshiftMaps.py)
    from ConvolveMaps import convolve_all_maps
    observed_maps_dir = os.path.join(workdir, "observed_maps")
    convolve_all_maps(
        spectral_image_file=os.path.join(render_dir, "stellar_spectral_image.fits"),
        wavelength_grid_file=os.path.join(render_dir, "wavelength_grid.fits"),
        maps_dir="./deshifted_maps",
        maps_wavefile="./reference_maps/cm_wavelengths.fits",
        savedir=observed_maps_dir,
        wvln_cntl=656.28, delta_wvln=5.0,
        plate_scale=5.9e-3
    )

    # --- 4. Measure centroid shifts ----------------------------------------
    from CentroidShifts import measure_centroid_shifts
    wavs, shifts = measure_centroid_shifts(
        cm_dir=observed_maps_dir,
        ref_dir="./reference_maps",
        ref_wavefile="./reference_maps/cm_wavelengths.fits",
        model_boundary=MODEL_BOUNDARY,
        plate_scale=PLATE_SCALE
    )

    if clean:
        shutil.rmtree(workdir, ignore_errors=True)

    return wavs, shifts


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------
def cost(theta, obs_wav, obs_shifts, mask=None):
    """
    theta = [omega_frac, inclination_deg, R_p / R_sun]
    """
    omega_frac, inclination, Rp_Rs = theta
    R_p = Rp_Rs * R_sun.value

    try:
        model_wav, model_shifts = run_forward_model(
            omega_frac, inclination, R_p, clean=True
        )
    except Exception as e:
        print(f"Forward model failed for {theta}: {e}")
        return 1e6

    # Interpolate model onto observed wavelength grid
    from scipy.interpolate import interp1d
    if model_shifts.ndim == 1:
        model_shifts = model_shifts[:, None]

    chi2 = 0.0
    for k in range(model_shifts.shape[1]):
        f = interp1d(model_wav, model_shifts[:, k],
                     bounds_error=False, fill_value=0.0)
        m = f(obs_wav)
        o = obs_shifts[:, k] if obs_shifts.ndim > 1 else obs_shifts
        if mask is not None:
            m = m[mask]
            o = o[mask]
        residual = m - o
        chi2 += np.nansum(residual**2)

    # mild regularisation to keep parameters physical
    chi2 += 0.1 * ((omega_frac - 0.9)**2 + ((inclination - 55)/10)**2)
    
    # ----- print the value so you can watch progress -----
    print(f"χ² = {chi2:10.4f}   "
          f"ω={omega_frac:.4f}  i={inclination:5.2f}°  Rp={Rp_Rs:.4f} R☉")
    return chi2


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------
def fit_stellar_parameters(obs_file=OBSERVED_FILE,
                          method="differential_evolution",
                          n_workers=1):
    # Load observed signal
    data = np.load(obs_file)
    obs_wav    = data[:, 0]
    obs_shifts = data[:, 1:]          # (N, n_components)

    # Wavelength mask (optional – focus on the line core)
    mask = (obs_wav > MODEL_BOUNDARY[0]) & (obs_wav < MODEL_BOUNDARY[1])

    # Parameter bounds
    #   omega_frac ∈ [0.7, 0.99]
    #   inclination ∈ [30, 80] deg
    #   R_p / R_sun ∈ [1.4, 1.9]
    bounds = [(0.70, 0.99), (30.0, 80.0), (1.40, 1.90)]

    if method == "differential_evolution":
        result = differential_evolution(
            cost,
            bounds=bounds,
            args=(obs_wav, obs_shifts, mask),
            workers=n_workers,
            popsize=12,
            mutation=0.7,
            recombination=0.5,
            seed=42,
            polish=True,
            updating="deferred",
            disp=True
        )
    else:  # local minimiser from a good starting guess
        x0 = [0.923, 57.2, 1.634]
        result = minimize(
            cost, x0,
            args=(obs_wav, obs_shifts, mask),
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 40, "disp": True}
        )

    best = result.x
    print("\n=== Best-fit parameters ===")
    print(f"  omega_frac   = {best[0]:.4f}")
    print(f"  inclination  = {best[1]:.2f} deg")
    print(f"  R_p          = {best[2]:.4f} R_sun")
    print(f"  chi2         = {result.fun:.3f}")

    # Save
    np.save(os.path.join(RESULTS_DIR, "best_fit_params.npy"), best)
    with open(os.path.join(RESULTS_DIR, "fit_summary.txt"), "w") as f:
        f.write(f"omega_frac  = {best[0]:.6f}\n")
        f.write(f"inclination = {best[1]:.4f} deg\n")
        f.write(f"R_p / Rsun  = {best[2]:.6f}\n")
        f.write(f"chi2        = {result.fun:.4f}\n")

    # Optional: plot data vs best model
    model_wav, model_shifts = run_forward_model(
        best[0], best[1], best[2]*R_sun.value, clean=False
    )
    plt.figure(figsize=(8, 4))
    for k in range(obs_shifts.shape[1]):
        plt.plot(obs_wav, obs_shifts[:, k], "k.", label="observed" if k==0 else None)
        plt.plot(model_wav, model_shifts[:, k], "-", label="model" if k==0 else None)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Centroid shift [mas]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fit_comparison.png"), dpi=200)

    return result



if __name__ == "__main__":
    result = fit_stellar_parameters(method="differential_evolution", n_workers=4)