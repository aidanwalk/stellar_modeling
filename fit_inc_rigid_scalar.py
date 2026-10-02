#!/usr/bin/env python
"""
Fit inclination + amplitude scale to the observed spectro-astrometric
centroid-shift signal (both axes).

Forward model chain:
  Stellar3DModel → StellarSpectralImage → ConvolveMaps → CentroidShifts
"""

import os
import sys
import shutil
import tempfile
import numpy as np
from scipy.optimize import differential_evolution
from astropy.constants import M_sun, R_sun
from astropy.io import fits
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OBSERVED_FILE = (
    "/home/arcadia/mysoft/gradschool/dissertation/papers/"
    "humu_2026/figures/centroid_shifts_major_axis/centroid_shifts.npy"
    # <-- replace with the 2-axis file if it has a different name
)
WORK_ROOT   = "./fit_workdir"
RESULTS_DIR = "./fit_results"
os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed stellar parameters
M_STAR = 1.791 * M_sun.value
T_EFF  = 7550.0
BETA   = 0.25
PA     = -62.7

MODEL_BOUNDARY = (654.940, 657.70)    # nm
PLATE_SCALE    = 5.9                  # mas / pixel

# ---------------------------------------------------------------------------
# Interactive plot (two panels, one per axis)
# ---------------------------------------------------------------------------
plt.ion()
fig_live, axes_live = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
obs_lines = []
mod_lines = []
for ax in axes_live:
    ol, = ax.plot([], [], "k.", ms=3, label="observed")
    ml, = ax.plot([], [], "r-", lw=1.5, label="model")
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    ax.set_ylabel("Shift [mas]")
    obs_lines.append(ol)
    mod_lines.append(ml)
axes_live[0].legend(loc="upper right")
axes_live[1].set_xlabel("Wavelength [nm]")
title_live = fig_live.suptitle("")
fig_live.tight_layout()

def update_live_plot(obs_wav, obs_shifts, model_wav, model_shifts, i, scale, chi2):
    """Update the interactive figure with the latest model (both axes)."""
    n_comp = obs_shifts.shape[1]
    for k in range(min(n_comp, 2)):
        obs_lines[k].set_data(obs_wav, obs_shifts[:, k])
        mod_lines[k].set_data(model_wav, model_shifts[:, k])
        axes_live[k].relim()
        axes_live[k].autoscale_view()
    title_live.set_text(f"i = {i:.2f}°   scale = {scale:.3f}   χ² = {chi2:.2f}")
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()

# ---------------------------------------------------------------------------
# Forward-model wrapper
# ---------------------------------------------------------------------------
def run_forward_model(omega_frac, inclination, R_p, workdir=None, clean=True):
    if workdir is None:
        workdir = tempfile.mkdtemp(dir=WORK_ROOT)
    else:
        os.makedirs(workdir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(__file__))
    from Stellar3DModel import Stellar3DModel

    model = Stellar3DModel(
        M_STAR, R_p, T_EFF,
        beta=BETA, omega_frac=omega_frac,
        N_grid=2**5, render_resolution=2**5, overfill=1.2,
        rigid=True
    )

    T_vol    = model.build_temperature_volume()
    g_vol    = model.build_gravity_volume()
    vrad_vol = model.build_radial_velocity_volume(inclination, PA)

    image_T    = model.render(T_vol,    inclination, PA, mode="surface",
                              project_vcam=True, vcam_derot_ang=101.34)
    image_g    = model.render(g_vol,    inclination, PA, mode="surface",
                              project_vcam=True, vcam_derot_ang=101.34)
    image_vrad = model.render(vrad_vol, inclination, PA, mode="surface",
                              threshold=1e-3, project_vcam=True, vcam_derot_ang=101.34)

    image_vrad /= 1e3
    image_g     = np.log10(image_g * 1e2)
    pixel_scale = model.get_pixel_scale(inclination, PA)

    render_dir = os.path.join(workdir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    def _write_rendered(name, data, bunit):
        hdu = fits.PrimaryHDU(data.astype(np.float32))
        hdu.header["BUNIT"]   = bunit
        hdu.header["PXSCALE"] = float(pixel_scale)
        hdu.header["PXUNIT"]  = "m/pixel"
        hdu.writeto(os.path.join(render_dir, name), overwrite=True)

    _write_rendered("rendered_temperature.fits",     image_T,    "K")
    _write_rendered("rendered_gravity.fits",         image_g,    "log10(cm/s^2)")
    _write_rendered("rendered_radial_velocity.fits", image_vrad, "km/s")

    from StellarSpectralImage import build_spectral_image
    build_spectral_image(
        render_dir,
        wave_cntl=656.28, delta_wave=20.0,
        spectra_dir="/home/arcadia/mysoft/gradschool/699_2/pheonix_templates",
        save=True, save_dir=render_dir,
    )

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

    from CentroidShifts import measure_centroid_shifts
    wavs, shifts = measure_centroid_shifts(
        cm_dir=observed_maps_dir,
        ref_dir="./reference_maps",
        ref_wavefile="./reference_maps/cm_wavelengths.fits",
        model_boundary=MODEL_BOUNDARY,
        plate_scale=PLATE_SCALE
    )

    # global sign flip to match observed convention
    # shifts *= -1.0
    # This offset is to correct for going from vacuum to air, 
    # bc i never applied it to the simulated spectral images. Really we 
    # should divide by the index of refraction of air = 1.000277
    wavs   -= 0.18

    if clean:
        shutil.rmtree(workdir, ignore_errors=True)

    return wavs, shifts          # shifts shape (N, n_comp)

# ---------------------------------------------------------------------------
# Cost function  (inclination + universal scale) – both axes
# ---------------------------------------------------------------------------
# global counter so we can see progress even if χ² printing is delayed
_cost_calls = 0

def cost(theta, obs_wav, obs_shifts, mask=None,
         omega_frac=0.923, Rp_Rs=1.634):
    global _cost_calls
    _cost_calls += 1

    inclination, scale = theta
    scale = np.clip(scale, 0.0, 1.0)
    R_p = Rp_Rs * R_sun.value

    # heartbeat – printed immediately, even if the forward model later crashes
    print(f"[{_cost_calls:3d}] starting  i={inclination:6.2f}°  scale={scale:.3f}",
          flush=True)

    try:
        model_wav, model_shifts = run_forward_model(
            omega_frac, float(inclination), R_p, clean=True
        )
    except Exception as e:
        print(f"[{_cost_calls:3d}] Forward model FAILED: {e}", flush=True)
        return 1e6

    # ensure 2-D
    if model_shifts.ndim == 1:
        model_shifts = model_shifts[:, None]
    if obs_shifts.ndim == 1:
        obs_shifts = obs_shifts[:, None]

    model_shifts = model_shifts * scale

    from scipy.interpolate import interp1d
    chi2 = 0.0
    n_comp = min(model_shifts.shape[1], obs_shifts.shape[1])

    for k in range(n_comp):
        f = interp1d(model_wav, model_shifts[:, k],
                     bounds_error=False, fill_value=np.nan)
        m = f(obs_wav)
        o = obs_shifts[:, k].copy()

        if mask is not None:
            m = m[mask]
            o = o[mask]

        good = np.isfinite(m) & np.isfinite(o)
        m = m[good]
        o = o[good]

        if len(m) == 0:
            print(f"[{_cost_calls:3d}] no overlapping points – χ²=1e6", flush=True)
            return 1e6

        residual = m - o
        chi2 += np.nansum(residual**2)

    # live plot
    try:
        update_live_plot(obs_wav, obs_shifts, model_wav, model_shifts,
                         inclination, scale, chi2)
    except Exception:
        pass   # never let the plot kill the fit

    print(f"[{_cost_calls:3d}] χ² = {chi2:10.4f}   i = {inclination:6.2f}°   scale = {scale:.3f}",
          flush=True)
    return chi2

# ---------------------------------------------------------------------------
# Fitting routine
# ---------------------------------------------------------------------------
def fit_inclination_and_scale(obs_file=OBSERVED_FILE,
                              omega_frac=0.923,
                              Rp_Rs=1.634,
                              bounds_i=(40.0, 70.0),
                              bounds_s=(0.1, 1.0)):
    data = np.load(obs_file)

    # wavelength conversion (km/s → nm) – keep consistent with your data
    obs_wav    = data[:, 0] #/ 3e5 * 656.28 + 656.28
    obs_shifts = data[:, 1:] #/ 1000.0          # all shift columns
    obs_wav    = obs_wav[::-1]
    obs_shifts = obs_shifts[::-1]

    if obs_shifts.ndim == 1:
        obs_shifts = obs_shifts[:, None]

    mask = (obs_wav > MODEL_BOUNDARY[0]) & (obs_wav < MODEL_BOUNDARY[1])

    result = differential_evolution(
        cost,
        bounds=[bounds_i, bounds_s],
        args=(obs_wav, obs_shifts, mask, omega_frac, Rp_Rs),
        popsize=10,
        mutation=0.7,
        recombination=0.5,
        seed=42,
        polish=True,
        updating="deferred",
        disp=True,
        workers=1,
    )

    best_i, best_s = result.x
    chi2_min = result.fun

    # approximate 1-σ errors from Δχ² = 1 contour
    di, ds = 2.0, 0.05
    i_grid = np.linspace(max(bounds_i[0], best_i-di), min(bounds_i[1], best_i+di), 9)
    s_grid = np.linspace(max(bounds_s[0], best_s-ds), min(bounds_s[1], best_s+ds), 9)
    chi2_map = np.zeros((len(i_grid), len(s_grid)))
    for ii, i in enumerate(i_grid):
        for ss, s in enumerate(s_grid):
            chi2_map[ii, ss] = cost([i, s], obs_wav, obs_shifts, mask,
                                    omega_frac, Rp_Rs)

    good = chi2_map <= chi2_min + 1.0
    if np.any(good):
        i_good = i_grid[np.any(good, axis=1)]
        s_good = s_grid[np.any(good, axis=0)]
        err_i = 0.5*(i_good.max()-i_good.min()) if len(i_good)>1 else di
        err_s = 0.5*(s_good.max()-s_good.min()) if len(s_good)>1 else ds
    else:
        err_i, err_s = di, ds

    print("\n=== Best-fit parameters ===")
    print(f"  inclination = {best_i:.3f} ± {err_i:.3f} deg")
    print(f"  scale       = {best_s:.3f} ± {err_s:.3f}")
    print(f"  χ²_min      = {chi2_min:.4f}")

    np.save(os.path.join(RESULTS_DIR, "best_fit_i_scale.npy"),
            np.array([best_i, err_i, best_s, err_s]))

    plot_final_fit(best_i, err_i, best_s, err_s,
                   obs_file, omega_frac, Rp_Rs)

    return best_i, err_i, best_s, err_s, result

# ---------------------------------------------------------------------------
# Final static plot (both axes)
# ---------------------------------------------------------------------------
def plot_final_fit(best_i, err_i, best_s, err_s,
                   obs_file, omega_frac, Rp_Rs,
                   savepath=None):
    data = np.load(obs_file)
    obs_wav    = data[:, 0] #/ 3e5 * 656.28 + 656.28
    obs_shifts = data[:, 1:] #/ 1000.0
    obs_wav    = obs_wav[::-1]
    obs_shifts = obs_shifts[::-1]
    if obs_shifts.ndim == 1:
        obs_shifts = obs_shifts[:, None]

    R_p = Rp_Rs * R_sun.value
    model_wav, model_shifts = run_forward_model(
        omega_frac, best_i, R_p, clean=False
    )
    if model_shifts.ndim == 1:
        model_shifts = model_shifts[:, None]
    model_shifts = model_shifts * best_s

    n_comp = min(obs_shifts.shape[1], model_shifts.shape[1], 2)
    fig, axes = plt.subplots(n_comp, 1, figsize=(8, 3.5*n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]

    for k in range(n_comp):
        axes[k].plot(obs_wav, obs_shifts[:, k], "k.", ms=3, label="observed")
        axes[k].plot(model_wav, model_shifts[:, k], "r-", lw=1.5,
                     label=f"model (i={best_i:.2f}°, s={best_s:.3f})")
        axes[k].axhline(0, color="grey", ls="--", lw=0.7)
        axes[k].set_ylabel(f"Shift axis {k+1} [mas]")
        if k == 0:
            axes[k].legend(loc="upper right")
    axes[-1].set_xlabel("Wavelength [nm]")
    fig.suptitle(f"i = {best_i:.2f} ± {err_i:.2f}°   scale = {best_s:.3f} ± {err_s:.3f}")
    fig.tight_layout()

    if savepath is None:
        savepath = os.path.join(RESULTS_DIR, "inclination_scale_fit.png")
    fig.savefig(savepath, dpi=200)
    print(f"Final plot saved to {savepath}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    best_i, err_i, best_s, err_s, res = fit_inclination_and_scale(
        omega_frac=0.923,
        Rp_Rs=1.634,
        bounds_i=(40.0, 70.0),
        bounds_s=(0.5, 1.0)
    )
    print(f"\nInclination = {best_i:.2f} ± {err_i:.2f}°")
    print(f"Scale       = {best_s:.3f} ± {err_s:.3f}")