
import os 
import numpy as np
from astropy.constants import M_sun, R_sun




def make_model():
    from Stellar3DModel import Stellar3DModel

    model = Stellar3DModel(
        M_star, R_p, T_eff, beta=beta, omega_frac=omega_frac, 
        N_grid=N_grid, render_resolution=render_resolution, overfill=grid_overfill, rigid=rigid
    )

    # Build the volumes we want to render
    T_vol = model.build_temperature_volume()
    g_vol = model.build_gravity_volume()
    v_rad_vol = model.build_radial_velocity_volume(inclination, position_angle)
    

    # Render images of the stellar surface for each volume
    image_T = model.render(T_vol, inclination, position_angle,
                            mode="surface", threshold=0.01, 
                            project_vcam=project_vcam, vcam_derot_ang=derot_ang)
    image_g = model.render(g_vol, inclination, position_angle,
                            mode="surface", threshold=0.01, 
                            project_vcam=project_vcam, vcam_derot_ang=derot_ang)
    image_vrad = model.render(v_rad_vol, inclination, position_angle,
                                mode="surface", threshold=1e-3, 
                                project_vcam=project_vcam, vcam_derot_ang=derot_ang)

    pixel_scale = model.get_pixel_scale(inclination, position_angle,
                                    project_vcam=project_vcam,
                                    vcam_derot_ang=derot_ang)
    
    
    print()
    print("Rendering complete.")
    print()
    print("v_rad stats inside star:")
    print(f"\tmin  = {np.nanmin(v_rad_vol/1e3):0.3f} km/s")
    print(f"\tmax  = {np.nanmax(v_rad_vol/1e3):0.3f} km/s")
    print(f"\tmean = {np.nanmean(v_rad_vol/1e3):0.6f} km/s")
    
    
    # -------------------------------------------------------------------------
    # We are pretty much done here. The rest of this code is for visualization
    # and data saving. 
    # -------------------------------------------------------------------------
    
    image_vrad /= 1e3  # convert to km/s for better visualization
    image_g = np.log10(image_g * 1e2)  # convert to log10(g/cm/s^2) for better visualization
    
    
    
    
    # Save rendered images as fits --------------------------------------------
    from astropy.io import fits
    fits.writeto(
        os.path.join(results_dir, 'rendered_temperature.fits'), 
        data = image_T.astype(np.float32), 
        header = fits.Header({
            'BUNIT': 'K',
            'PXSCALE': pixel_scale,  # meters per pixel
            'PXUNIT': 'm/pixel',
            'COMMENT': 'Rendered temperature image from Stellar3DModel',
            'COMMENT': f'Model parameters: M={M_star:.2e} kg, R_p={R_p:.2e} m, T_eff={T_eff} K, beta={model.beta}, omega_frac={omega_frac}, inclination={inclination} deg, PA={position_angle} deg',
        }),
        overwrite=True
    )
    
    fits.writeto(
        os.path.join(results_dir, 'rendered_gravity.fits'), 
        data = image_g.astype(np.float32), 
        header = fits.Header({
            'BUNIT': 'log10(cm/s^2)',
            'PXSCALE': pixel_scale,  # meters per pixel
            'PXUNIT': 'm/pixel',
            'COMMENT': 'Rendered gravity image from Stellar3DModel (log10 scale)',
            'COMMENT': f'Model parameters: M={M_star:.2e} kg, R_p={R_p:.2e} m, T_eff={T_eff} K, beta={model.beta}, omega_frac={omega_frac}, inclination={inclination} deg, PA={position_angle} deg',
        }), 
        overwrite=True
    )
    
    fits.writeto(
        os.path.join(results_dir, 'rendered_radial_velocity.fits'), 
        data = image_vrad.astype(np.float32), 
        header = fits.Header({
            'BUNIT': 'km/s',
            'PXSCALE': pixel_scale,  # meters per pixel
            'PXUNIT': 'm/pixel',
            'COMMENT': 'Rendered radial velocity image from Stellar3DModel',
            'COMMENT': f'Model parameters: M={M_star:.2e} kg, R_p={R_p:.2e} m, T_eff={T_eff} K, beta={model.beta}, omega_frac={omega_frac}, inclination={inclination} deg, PA={position_angle} deg',
        }),
        overwrite=True
    )
    return




if __name__ == "__main__":
    # # STELLAR PARAMETERS (Alderamin) ---------------------------------------------
    M_star = 2.00 * M_sun.value    # stellar mass in kg
    R_p = 2.175 * R_sun.value       # polar radius of the star in meters
    T_eff = 7550.0                  # K, effective temperature (used only for normalizing C_omega)
    omega_frac = 0.83              # Fraction of critical rotation
    inclination = 88.2              # degrees, 0 = pole-on, 90 = edge-on
    position_angle = 17.2          # degrees, 0 = north up, positive eastward
    beta = 0.25                     # UNTESTED FOR VALUES != 0.25; gravity darkening exponent (von Zeipel)
    project_vcam = True             # Whether to project on-sky or project on vcam (False for on-sky, True for vcam)
    derot_ang = -98.044             # degrees, angle to rotate the vcam projection to match on-sky orientation (only used if project_vcam=True)
    rigid = False                   # Whether to treat the star as a rigid rotator (no gravity darkening)    
    # -------------------------------------------------------------------------
    
    # MODEL RESOLUTION --------------------------------------------------------
    N_grid = 2**5                   # 3D grid resolution (N x N x N)
    render_resolution = 2**6        # Final rendered image resolution
    grid_overfill = 1.1            # How much larger the grid should be than the star's polar radius (in units of R_p).
    # -------------------------------------------------------------------------
    
    
    
    results_dir = './renders/Alderamin'
    plots_dir = './plots/Alderamin'
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    make_model()