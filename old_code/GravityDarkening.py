"""
This script is for modeling the effects of gravitational darkening of a star.

Main Goal:
Make an image of the star taking into account the effects of gravitational 
darkening. This is a 3D image (x, y, lambda), since gravity darkening is a
function of position on stellar limb (x,y) and Temperature (relative intensity; 
which is a function of wavelength (lambda).

The output of the code should be a 3D array of the star's relative intensity
as a function of (x,y,lambda) for given stellar parameters and on-sky orientation.



How-to:
1. Create a 3D image of the star (x, y, z) whose values are surface 
   temperatures (T) at each point on the star's surface.
2. Rotate the image to the desired on-sky orientation (given by inclination and
   position angle) to produce a 2D image of the star (x, y) with values of T at 
   each point on the star's surface. 
3. Convert the 2D image of T to a 3D image of relative intensity (x, y, lambda)
   using the planck function to convert T to intensity at each wavelength 
   (lambda) of interest.
4. Output the 3D image of relative intensity (x, y, lambda) for use in other
   codes
   

"""



import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.spatial import KDTree
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import binary_fill_holes




from scipy.constants import sigma, G
from astropy.constants import M_sun, R_sun

        
    
    




def omega_critical(M_star, R_p):
    """
    Calculate the critical angular rotation rate
    
    Parameters:
    -----------
    M_star : float or ndarray
        Mass of the star in kg
    R_p : float or ndarray
        Polar radius of the star in meters
    
    Returns 
    -------
    omega_crit : float or ndarray
        Critical angular rotation rate in radians per second
    """
    return (8 * G * M_star / (27 * R_p**3))**0.5



def model_R(theta, R_p, omega_frac):
    """
    Calculate the radius of the star as a function of colatitude (theta)
    Equation taken from McGill, Sigut, Jones 2011 ApJ
    
    Parameters:
    -----------
    theta : float or ndarray
        Colatitude in radians (0 at pole, pi/2 at equator)
    R_p : float
        Polar radius of the star in meters
    omega_frac : float
        Angular rotation rate as a fraction of critical rotation
    
    
    Returns:
    --------
    R : float or ndarray
        Radius of the star at colatitude theta in meters
    """
    A = -3 * R_p / (omega_frac * np.sin(theta))
    B = np.arccos(omega_frac * np.sin(theta))
    R = A * np.cos( (B+4*np.pi) / 3 )
    if np.isscalar(theta):
        if theta % np.pi == 0:
            R = R_p
    else:
        R[theta%np.pi == 0] = R_p # Set the radius at the poles to be R_p
    return R





def gravitational_acceleration(theta, phi, radius, M_star, omega):
    """
    Calculate the gravitational acceleration at each point on the star's surface
    as a function of colatitude (theta) and radius.
    
    Formula is from McGill, Sigut, Jones 2011 ApJ, 
    
    Parameters:
    -----------
    theta : float or ndarray
        Colatitude in radians (0 at pole, pi/2 at equator)
    R : float or ndarray
        Radius of the star at colatitude theta in meters
    M_star : float
        Mass of the star in kg
    
    Returns:
    --------
    g : float or ndarray
        Gravitational acceleration at colatitude theta in m/s^2
    """
    r_mag = (omega**2 * radius * np.sin(theta)**2 - G*M_star/radius**2)
    r_hat = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    
    theta_mag = (omega**2 * radius * np.sin(theta) * np.cos(theta))
    theta_hat = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    
    g = r_mag * r_hat + theta_mag * theta_hat
    return g





def compute_von_zeipel_const(theta, phi, R, g, T_eff, beta=1/4):
    """
    Compute von Zeipel's constant (C_omega).
    Derived by enforcing luminosity conservation:
        L_star = sigma * C_omega^4 * integral(g_eff dA)
    So:
        C_omega = (L_star / (sigma * integral(g_eff dA)))^(1/4)
    """
       
    g_mag = np.linalg.norm(g, axis=0)  # shape (N_phi, N_theta)

    # Area element for oblate spheroid: R^2 * sin(theta) dtheta dphi
    dA = R**2 * np.sin(theta)
    surface_area = np.abs(np.trapezoid(np.trapezoid(dA, phi, axis=0), theta[0, :], axis=0))
    L_star = sigma * T_eff**4 * surface_area

    # Integrate g * dA over the surface using physical coordinates
    integrand = g_mag**(4*beta) * dA
    surface_integral = np.abs(np.trapezoid(
        np.trapezoid(integrand, phi, axis=0), theta[0, :], axis=0
    ))

    C_omega = (L_star / (sigma * surface_integral))**beta
    return C_omega





def project_on_sky(theta, phi, R, inclination, position_angle, color=None):
    """
    Project the 3D coordinates of the star's surface onto the sky plane given 
    the inclination and position angle of the star. This will produce a 2D 
    image of the star as it would appear on the sky, with the values at each 
    point corresponding to the colatitude (theta) at that point on the star's 
    surface.
    """
    # tilt by the negative of the inclination to get the correct orientation on the sky
    inclination = -inclination
    
    # project to a 2D plane given inclination and position angle
    PA_rotation_matrix = np.array([[np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
                                   [np.sin(np.radians(position_angle)), np.cos(np.radians(position_angle)), 0],
                                   [0, 0, 1]])
    inclination_rotation_matrix = np.array([[1, 0, 0],
                                           [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
                                           [0, np.sin(np.radians(inclination)), np.cos(np.radians(inclination))]])
    projection_matrix = PA_rotation_matrix @ inclination_rotation_matrix
    
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    star_coords = np.array([x.flatten(), y.flatten(), z.flatten()])
    projected_coords = projection_matrix @ star_coords
    x_proj = projected_coords[0, :].reshape(theta.shape)
    y_proj = projected_coords[1, :].reshape(theta.shape)
    z_proj = projected_coords[2, :].reshape(theta.shape)
    
    visible = z_proj > 0 # Only keep points on the visible hemisphere
    if color is None:
        color = theta
        
    return x_proj[visible], y_proj[visible], z_proj[visible], color[visible]






def make_image(x, y, color, shape=(200, 200)):

    x = np.asarray(x)
    y = np.asarray(y)
    color = np.asarray(color)

    x_edges = np.linspace(np.min(x), np.max(x), shape[0] + 1)
    y_edges = np.linspace(np.min(y), np.max(y), shape[1] + 1)

    img, _, _, _ = binned_statistic_2d(
        x, y, color, statistic='mean', bins=[x_edges, y_edges]
    )
    counts, _, _, _ = binned_statistic_2d(
        x, y, color, statistic='count', bins=[x_edges, y_edges]
    )

    img = img.T
    counts = counts.T

    # Build the stellar disk mask: pixels that were sampled, then fill
    # interior holes to get the full disk footprint
    sampled = counts > 0
    disk_mask = binary_fill_holes(sampled)

    # Fill interior holes using nearest-neighbor interpolation over valid pixels
    interior_holes = disk_mask & ~sampled
    if interior_holes.any():
        valid_coords = np.argwhere(sampled)
        valid_values = img[sampled]
        interp = NearestNDInterpolator(valid_coords, valid_values)
        img[interior_holes] = interp(np.argwhere(interior_holes))

    # Set everything outside the stellar disk to NaN
    img[~disk_mask] = np.nan

    return img





import numpy as np
from scipy.interpolate import RegularGridInterpolator


def render_object(x, y, z, c, R, resolution=256, mode="surface",
                  threshold=0.01, n_samples=None):
    """
    Render a 3D greyscale volume as a 2D image using ray casting.

    For every output pixel a ray is marched through the volume along the
    camera depth axis.  The field is sampled with trilinear interpolation
    (RegularGridInterpolator), so there are no holes and occlusion is
    handled exactly the same way matplotlib handles its 3-D plots: front
    geometry hides rear geometry.

    Parameters
    ----------
    x, y, z : 1-D array-like
        Coordinate axes of the rectilinear grid.
        `c` must satisfy  c.shape == (len(x), len(y), len(z)).
    c : ndarray, shape (Nx, Ny, Nz)
        Greyscale intensity at every grid point.  Values should be >= 0;
        the output is not renormalised.
    R : array-like, shape (3, 3)
        Rotation (camera) matrix.  Applied as  p_cam = R @ p_world, so
        the *rows* of R are the world-space directions of the camera's
        X, Y, Z axes respectively.  Must be orthonormal.
    resolution : int
        Width and height of the square output image in pixels.
    mode : {"surface", "mip", "volume"}
        Compositing rule along each ray:

        "surface"  – first-hit ray casting.  Returns the intensity of the
                     first sample along the ray that exceeds `threshold`.
                     Produces hard, opaque surfaces — closest to a
                     matplotlib 3-D surface plot.
        "mip"      – maximum-intensity projection.  Returns the brightest
                     sample along the ray, regardless of depth.
        "volume"   – front-to-back alpha compositing.  Treats `c` as an
                     emission+absorption density; produces translucent /
                     X-ray-style images.
    threshold : float
        Surface hit threshold used only when mode="surface".
    n_samples : int or None
        Number of evenly-spaced samples along each ray.  Default is
        3× the length of the longest grid axis — enough to guarantee at
        least one sample per voxel even after rotation.

    Returns
    -------
    image : ndarray, shape (resolution, resolution), dtype float64
        Rendered greyscale image.  Row 0 is the top of the image
        (maximum projected-Y value in camera space).  Empty pixels are 0.
    """
    R = np.asarray(R, dtype=float)
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if c.shape != (len(x), len(y), len(z)):
        raise ValueError(
            f"c.shape {c.shape} does not match grid dimensions "
            f"({len(x)}, {len(y)}, {len(z)})"
        )

    # ------------------------------------------------------------------
    # 1.  Trilinear interpolator for the scalar field
    # ------------------------------------------------------------------
    interp = RegularGridInterpolator(
        (x, y, z), c,
        method="linear",
        bounds_error=False,
        fill_value=0.0,       # rays that leave the grid contribute 0
    )

    # ------------------------------------------------------------------
    # 2.  Camera-space axes (rows of R expressed in world coords)
    #
    #     p_world = R^T @ p_cam
    #     A pixel at camera position (u, v) defines a ray:
    #       p_world(t) = u·R[0] + v·R[1] + t·R[2]
    #     where t is depth along the camera Z axis.
    # ------------------------------------------------------------------
    cam_x = R[0]   # world-space direction of camera horizontal axis
    cam_y = R[1]   # world-space direction of camera vertical axis
    cam_z = R[2]   # world-space direction of camera depth axis (ray dir.)

    # ------------------------------------------------------------------
    # 3.  Find the bounding box of the volume in camera space
    #     (project all 8 corners of the AABB)
    # ------------------------------------------------------------------
    corners = np.array([
        [xi, yi, zi]
        for xi in (x[0], x[-1])
        for yi in (y[0], y[-1])
        for zi in (z[0], z[-1])
    ])                               # shape (8, 3)
    corners_cam = corners @ R.T      # shape (8, 3) — same as (R @ corners.T).T

    u_min, u_max = corners_cam[:, 0].min(), corners_cam[:, 0].max()
    v_min, v_max = corners_cam[:, 1].min(), corners_cam[:, 1].max()
    t_min, t_max = corners_cam[:, 2].min(), corners_cam[:, 2].max()

    # Square pixel grid with uniform aspect ratio + 2 % margin
    u_ctr = (u_min + u_max) / 2
    v_ctr = (v_min + v_max) / 2
    half  = max(u_max - u_min, v_max - v_min) / 2 * 1.02

    u_vals = np.linspace(u_ctr - half, u_ctr + half, resolution)
    v_vals = np.linspace(v_ctr - half, v_ctr + half, resolution)

    # ------------------------------------------------------------------
    # 4.  Depth samples — default 3× longest axis for full coverage
    # ------------------------------------------------------------------
    if n_samples is None:
        n_samples = int(3 * max(len(x), len(y), len(z)))
    t_vals = np.linspace(t_min, t_max, n_samples)

    # ------------------------------------------------------------------
    # 5.  Build the pixel-centre grid once (broadcast-friendly)
    #     UU[i, j] = u_vals[j]  (column index → horizontal)
    #     VV[i, j] = v_vals[i]  (row    index → vertical)
    # ------------------------------------------------------------------
    UU, VV = np.meshgrid(u_vals, v_vals)   # each shape (res, res)

    # Precompute the ray-origin field (depth-independent part)
    # origin[i, j] = UU[i,j] * cam_x + VV[i,j] * cam_y   shape (res, res, 3)
    origin = (UU[..., np.newaxis] * cam_x +
              VV[..., np.newaxis] * cam_y)     # (res, res, 3)

    # ------------------------------------------------------------------
    # 6.  Ray march and composite
    # ------------------------------------------------------------------
    image = np.zeros((resolution, resolution), dtype=float)

    if mode == "surface":
        hit_mask  = np.zeros((resolution, resolution), dtype=bool)
        hit_color = np.zeros((resolution, resolution), dtype=float)

        for t in t_vals:
            if hit_mask.all():
                break
            pts  = origin + t * cam_z           # (res, res, 3)
            vals = interp(pts)                   # (res, res)

            new_hit = ~hit_mask & (vals > threshold)
            hit_color[new_hit] = vals[new_hit]
            hit_mask |= new_hit

        image = hit_color

    elif mode == "mip":
        for t in t_vals:
            pts  = origin + t * cam_z
            vals = interp(pts)
            np.maximum(image, vals, out=image)

    elif mode == "volume":
        # Front-to-back alpha compositing
        # α(t) = 1 − exp(−σ · Δt),  σ = c(t)
        dt           = (t_max - t_min) / n_samples
        transmittance = np.ones((resolution, resolution), dtype=float)

        for t in t_vals:
            pts   = origin + t * cam_z
            sigma = interp(pts)                  # emission + absorption density
            alpha = 1.0 - np.exp(-sigma * dt)
            image         += transmittance * alpha * sigma
            transmittance *= 1.0 - alpha

    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'surface', 'mip', or 'volume'.")

    # Row 0 = top of image (largest v), flip vertically
    return image




# %%

if __name__ == "__main__":
    
    # DEFINE ALTAIR PARAMETERS ------------------------------------------------
    M_star = 1.791 * M_sun.value                    # Mass in kg
    R_p = 1.634 * R_sun.value                       # Polar radius in meters
    T_eff = 7550                                    # effective temperature of Altair
    beta = 0.25                                     # Gravity darkening exponent
    omega_frac = 0.923                              # Angular rotation rate as a fraction of critical rotation
    inclination = 57.2                              # Inclination in degrees
    position_angle = -61.8                          # Position angle in degrees
    # -------------------------------------------------------------------------
    
    
    
    # Define the coordinate grid for the star's surface in equal area bands
    N = 64
    u = np.linspace(-1, 1, N)
    theta = np.arccos(u)
    phi = np.linspace(0, 2*np.pi, N)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Radius of star as a function of equator
    R_grid = model_R(theta_grid, R_p, omega_frac)
    
    # Compute the gravitational acceleration at each point on the star's surface
    omega = omega_frac * omega_critical(M_star, R_p)
    g = gravitational_acceleration(theta_grid, phi_grid, R_grid, M_star, omega)
    g_mag = np.linalg.norm(g, axis=0) # Get the magnitude of the gravitational acceleration at each point on the star's surface
    
    
    # Use Von-Zeipels Thm to compute the local temperature at each point on the
    # star's surface
    C_omega = compute_von_zeipel_const(theta_grid, phi_grid, R_grid, g, T_eff)
    T_grid = C_omega * g_mag**beta # Local temperature at
    
    print("von Zeipel constant:", C_omega)
    
    
    
    
    import matplotlib.pyplot as plt
    plt.close('all')
    
    # Plot star colored by colatitude (theta) ---------------------------------
    x, y, z, color = project_on_sky(
        theta_grid, phi_grid, R_grid, 
        inclination, 
        position_angle, 
        color=theta_grid
    )
    
    plt.figure()
    plt.scatter(x, y, c=color, cmap='viridis', s=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Altair Modeled Shape on Sky')
    plt.colorbar(label='Colatitude (radians)')
    plt.axis('equal')
    plt.savefig('Altair_colatitude.png', dpi=300)
    # -------------------------------------------------------------------------
    
    
    # Plot star colored by gravitational acceleration (g) ---------------------
    x, y, z, color = project_on_sky(
        theta_grid, phi_grid, R_grid, 
        inclination, 
        position_angle, 
        color=g_mag
    )
    # color = np.log10(color * 1e2) # Convert to log10(g/cm/s^2) for better visualization
    plt.figure()
    plt.scatter(x, y, c=color, cmap='viridis', s=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Altair Modeled g')
    plt.colorbar(label=r'g ($m/s^2$)')
    plt.axis('equal')
    plt.savefig('Altair_g.png', dpi=300)
    # -------------------------------------------------------------------------
    
    
    # Plot star colored by temperature (T) ------------------------------------
    x, y, z, color = project_on_sky(
        theta_grid, phi_grid, R_grid, 
        inclination, 
        position_angle, 
        color=T_grid
    )
    
    plt.figure()
    plt.scatter(x, y, c=color, cmap='viridis', s=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Altair Modeled Surface Temperature')
    plt.colorbar(label=r'T (K)')
    plt.axis('equal')
    plt.savefig('Altair_T.png', dpi=300)
    # -------------------------------------------------------------------------
    
    # %%
    # Create variables for render volume function
    x = np.linspace(-2*R_p, 2*R_p, N)
    y = np.linspace(-2*R_p, 2*R_p, N)
    z = np.linspace(-2*R_p, 2*R_p, N)
    c = np.zeros((N, N, N)) * np.nan
    
        
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Convert (x,y,z) to (theta, phi, R)
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                if r == 0:
                    continue
                theta = np.arccos(z[k]/r)
                phi = np.arctan2(y[j], x[i])
                
                # Get the corresponding temperature at this point on the star's surface
                R_surface = model_R(theta, R_p, omega_frac)
                if r <= R_surface:
                    g_surface = gravitational_acceleration(theta, phi, R_surface, M_star, omega)
                    g_mag_surface = np.linalg.norm(g_surface)
                    # C_omega_surface = compute_von_zeipel_const(theta, phi, R_surface, g_surface, T_eff)
                    T_surface = C_omega * g_mag_surface**beta
                    c[i,j,k] = T_surface
    
    
    
    
    
    # Create the rotation matrix for the desired inclination and position angle
    PA_rotation_matrix = np.array([[np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
                                   [np.sin(np.radians(position_angle)), np.cos(np.radians(position_angle)), 0],
                                   [0, 0, 1]])
    inclination_rotation_matrix = np.array([[1, 0, 0],
                                           [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
                                           [0, np.sin(np.radians(inclination)), np.cos(np.radians(inclination))]])
    R = PA_rotation_matrix @ inclination_rotation_matrix
    
    
    
    # %%
    
    
    image = render_object(x, y, z, c, R, threshold=0.01, mode="surface", resolution=512)
    image[image == 0] = np.nan # Set empty pixels to NaN for better visualization
    
    # %%
    plt.figure()
    plt.imshow(image, origin='lower', cmap='viridis')
    plt.xlabel('X (px)')
    plt.ylabel('Y (px)')
    plt.xlim(106, 406)
    plt.ylim(106, 406)
    plt.title('Binned Image Altair $T_{surf}$')
    plt.colorbar(label=r'T (K)')
    # plt.axis('equal')
    plt.savefig('Altair_T_binned_image.png', dpi=300)
    
    plt.close('all')
    