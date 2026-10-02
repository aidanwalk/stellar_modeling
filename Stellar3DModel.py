"""
Stellar3DModel is a class for modeling a rapidly rotating star in 3D, 
including its temperature and gravity fields, and rendering an image from any
on-sky orientation (position angle and inclination). The output of this code 
is is a rendered "on-sky" image of the star, whose pixels values represent 
surface temperature, gravitational acceleration, or radial velocity. 


The model is based on this paper:
    McGill, Sigut, Jones 2011 ApJ
    The Thermal Structure of Gravitationally-Darkened Classical Be Star Disks. 
    


"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import sigma, G
from astropy.constants import M_sun, R_sun


class Stellar3DModel:
    def __init__(self, M_star, R_p, T_eff, beta=0.25, omega_frac=None, v_rot=None,
                 N_grid=256, render_resolution=512, overfill=2.0, rigid=False):
        """
        Initialize the stellar model.

        Parameters
        ----------
        M_star : float
            Stellar mass (kg)
        R_p : float
            Polar radius (m)
        T_eff : float
            Effective temperature (K) — used only for normalizing C_ω
        beta : float, default 0.25
            Gravity darkening exponent (von Zeipel)
        omega_frac : float, default 0.0
            Rotation rate as fraction of critical rotation. Provide either `omega_frac` or `v_rot`.
        v_rot : float, default None
            Equatorial rotational velocity (m/s). Provide either `v_rot` or `omega_frac`.
        N_grid : int, default 256
            Resolution for the 3D volume grid (x, y, z)
        render_resolution : int, default 512
            Resolution of the final rendered image
        overfill : float, default 2.0
            How much larger the grid should be than the star's polar radius (in units of R_p).
        rigid : bool, default False
            If True, the star is treated as a rigid rotator (no gravity darkening).
        """
        self.M_star = M_star
        self.R_p = R_p
        self.T_eff = T_eff
        self.beta = beta
        self.N = N_grid
        self.render_resolution = render_resolution
        self.grid_half_width = overfill * R_p  # grid extends to ±2 polar radii in each dimension
        self.rigid = rigid
        
        assert (omega_frac is not None) or (v_rot is not None), "Provide either omega_frac or v_rot"
        if v_rot is not None:
            self.omega = v_rot / self.R_p
            self.omega_frac = self.omega / self._omega_critical()
        else:
            self.omega_frac = omega_frac
            self.omega = self.omega_frac * self._omega_critical()
            
        self.C_omega = None

        # Grid will be built on first call to build_*
        self.x = None
        self.y = None
        self.z = None
        self.R_grid = None
        self.mask = None
        self.theta_grid = None
        self.phi_grid = None
        self.R_surface = None
        self.inside = None
        
        self._ensure_grid()



    # ------------------------------------------------------------------
    # Physics helpers (fully vectorized)
    # ------------------------------------------------------------------
    def _omega_critical(self):
        """Critical angular velocity"""
        return (8 * G * self.M_star / (27 * self.R_p**3))**0.5



    def model_R(self, theta):
        """
        Calculate the radius of the star as a function of colatitude (theta)
        Equation taken from McGill, Sigut, Jones 2011 ApJ
        
        Parameters:
        -----------
        theta : array-like
            Colatitude in radians (0 at pole, π/2 at equator)
            
        Returns:
        --------
        R : array-like
            Radius of the star at each colatitude
        """
        
        if self.rigid:
            # For a rigid rotator, the radius is constant (no oblateness)
            return np.full_like(theta, self.R_p)
        
        theta = np.asarray(theta, dtype=float)
        A = -3 * self.R_p / (self.omega_frac * np.sin(theta))
        B = np.arccos(self.omega_frac * np.sin(theta))
        R = A * np.cos((B + 4 * np.pi) / 3)

        # Fix poles (sin(theta) ≈ 0)
        pole_mask = np.abs(np.sin(theta)) < 1e-10
        R[pole_mask] = self.R_p
        return R



    def gravitational_acceleration(self, theta, phi, radius):
        """
        Effective gravitational acceleration vector (..., 3)
        
        Calculate the gravitational acceleration at each point on the star's surface
        as a function of theta, phi, and radius. This includes both the 
        Newtonian gravity and the centrifugal force
        
        Formula is from McGill, Sigut, Jones 2011 ApJ, 
        
        Parameters:
        -----------
        theta : array-like
            Colatitude in radians (0 at pole, π/2 at equator)
        phi : array-like
            Azimuthal angle in radians
        radius : array-like
            Radius at each (theta, phi) point
            
        Returns:
        --------
        g : array-like
            Gravitational acceleration vector at each point (shape (..., 3))
        """
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        radius = np.asarray(radius, dtype=float)

        r_mag = (self.omega**2 * radius * np.sin(theta)**2 -
                 G * self.M_star / radius**2)
        theta_mag = self.omega**2 * radius * np.sin(theta) * np.cos(theta)

        r_hat = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ], axis=-1)

        theta_hat = np.stack([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ], axis=-1)

        g = r_mag[..., None] * r_hat + theta_mag[..., None] * theta_hat
        return g



    def compute_von_zeipel_const(self, theta_grid, phi_grid, R_grid):
        """
        Compute von Zeipel's constant (C_omega).
        Derived by enforcing luminosity conservation:
            L_star = sigma * surface_area * T**4 = sigma * C_omega * integral(g_eff dA)
        So:
            C_omega = (L_star / (sigma * integral(g_eff dA)))
        """
        g = self.gravitational_acceleration(theta_grid, phi_grid, R_grid)
        g_mag = np.linalg.norm(g, axis=-1)

        dA = R_grid**2 * np.sin(theta_grid)
        # Compute the surface area of the star for luminosity calculation
        surface_area = np.abs(np.trapezoid(
            np.trapezoid(dA, phi_grid, axis=0), theta_grid[0, :], axis=0))
        L_star = sigma * self.T_eff**4 * surface_area

        # Integrate over g_eff
        integrand = g_mag * dA
        surface_integral = np.abs(np.trapezoid(
            np.trapezoid(integrand, phi_grid, axis=0), theta_grid[0, :], axis=0))
        C_omega = (L_star / (sigma * surface_integral))
        
        return C_omega




    # ------------------------------------------------------------------
    # Volume builders (vectorized, no loops)
    # ------------------------------------------------------------------
    def _ensure_grid(self):
        """Build the 3D rectilinear grid once (shared by T and g volumes)"""
        if self.x is not None:
            return

        lim = self.grid_half_width
        self.x = np.linspace(-lim, lim, self.N)
        self.y = np.linspace(-lim, lim, self.N)
        self.z = np.linspace(-lim, lim, self.N)

        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        self.R_grid = np.sqrt(X**2 + Y**2 + Z**2)
        self.mask = self.R_grid > 1e-12

        theta = np.full_like(self.R_grid, np.nan)
        phi = np.full_like(self.R_grid, np.nan)
        theta[self.mask] = np.arccos(Z[self.mask] / self.R_grid[self.mask])
        phi[self.mask] = np.arctan2(Y[self.mask], X[self.mask])

        self.theta_grid = theta
        self.phi_grid = phi
        
        self.R_surface = self.model_R(self.theta_grid)          # NaNs outside mask are harmless
        self.inside = (self.R_grid <= self.R_surface) & self.mask



    def _build_scalar_volume(self, scalar_func):
        """Private helper: evaluates any scalar field only inside the star"""
        self._ensure_grid()
        scalar = np.full_like(self.R_grid, np.nan)
        theta_in = self.theta_grid[self.inside]
        phi_in = self.phi_grid[self.inside]
        R_surf_in = self.R_surface[self.inside]
        scalar[self.inside] = scalar_func(theta_in, phi_in, R_surf_in)
        return scalar




    def build_temperature_volume(self):
        """
        Build 3D temperature volume (x, y, z) → T (K).
        Returns the array 
        C_ω is computed once on a coarse surface grid.
        
        The temperature at each point is given by von Zeipel's law:
            T(theta, phi) = C_ω * |g_eff(theta, phi)|^beta
        where g_eff is the effective gravity vector at the surface point (theta, phi).
        """
        self._ensure_grid()

        if self.C_omega is None:
            # Coarse surface grid for C_ω (sufficient accuracy, fast)
            N_surf = 200
            u = np.linspace(-1, 1, N_surf)
            theta_surf = np.arccos(u)
            phi_surf = np.linspace(0, 2 * np.pi, 2 * N_surf)
            theta_surf_grid, phi_surf_grid = np.meshgrid(theta_surf, phi_surf)
            R_surf_grid = self.model_R(theta_surf_grid)

            self.C_omega = self.compute_von_zeipel_const(
                theta_surf_grid, phi_surf_grid, R_surf_grid)
            # print(f"von Zeipel constant C_ω = {self.C_omega:.6e}")

        
        if self.rigid:
            # For rigid rotation, temperature is uniform (no gravity darkening)
            def scalar_func(theta, phi, R_surf):
                return self.T_eff
            
            return self._build_scalar_volume(scalar_func)
        
    
        def scalar_func(theta, phi, R_surf):
            g_vec = self.gravitational_acceleration(theta, phi, R_surf)
            g_mag = np.linalg.norm(g_vec, axis=-1)
            return (self.C_omega * g_mag)**self.beta

        return self._build_scalar_volume(scalar_func)




    def build_gravity_volume(self):
        """
        Build 3D gravitational acceleration magnitude volume (x, y, z) → |g| (m/s^2).
        Returns the array.
        units are in m/s^2
        """
        self._ensure_grid()
        
        # if self.rigid:
        #     # For rigid rotation, gravity is uniform (no gravity darkening)
        #     def scalar_func(theta, phi, R_surf):
        #         g_vec = self.gravitational_acceleration(theta, phi, R_surf)
        #         return np.linalg.norm(g_vec, axis=-1)
            
        #     return self._build_scalar_volume(scalar_func)

        def scalar_func(theta, phi, R_surf):
            g_vec = self.gravitational_acceleration(theta, phi, R_surf)
            return np.linalg.norm(g_vec, axis=-1)

        return self._build_scalar_volume(scalar_func)

    
    
    
    def build_rotational_velocity_volume(self):
        """
        Build 3D rotational velocity magnitude volume (x, y, z) → v_rot.
        Returns the array.
        """
        self._ensure_grid()

        def scalar_func(theta, phi, R_surf):
            return self.omega * R_surf * np.sin(theta)

        return self._build_scalar_volume(scalar_func)
    
    
    
    def build_radial_velocity_volume(self, inclination, position_angle):
        """
        Build 3D radial velocity volume (x, y, z) → v_radial (m/s).
        
        This is the line-of-sight (observer's Z) component of the rotational 
        velocity field after applying inclination and position angle.
        Positive = receding (redshift), negative = approaching (blueshift).
        
        Returns the volume with NaNs outside the star.
        """
        self._ensure_grid()

        # Rotation matrix: star frame → observer frame (same as render())
        PA_rotation_matrix = np.array([
            [np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
            [np.sin(np.radians(position_angle)),  np.cos(np.radians(position_angle)), 0],
            [0, 0, 1]
        ])
        inclination_rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
            [0, np.sin(np.radians(inclination)),  np.cos(np.radians(inclination))]
        ])
        R = PA_rotation_matrix @ inclination_rotation_matrix   # (3, 3)

        def scalar_func(theta, phi, R_surf):
            # Rotational velocity magnitude at this latitude
            v_eq = self.omega * R_surf * np.sin(theta)          # scalar, shape (...)

            # Rotational velocity direction in star's frame (prograde)
            # v = ω × r  → tangential vector in equatorial plane
            v_unit = np.stack([
                -np.sin(phi),      # x-component
                np.cos(phi),      # y-component
                np.zeros_like(phi)  # z-component (no vertical motion from rotation)
            ], axis=-1)   # shape (..., 3)

            v_rot_star = v_eq[..., None] * v_unit   # (..., 3)

            # Transform to observer frame
            v_rot_obs = v_rot_star @ R.T            # (..., 3)

            # Line-of-sight component = component along observer's line of sight.
            # After the rotation matrix used in your render_object(), 
            # this is the **third component** (Z in camera space).
            return v_rot_obs[..., 2]

        return self._build_scalar_volume(scalar_func)
    
    
    
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, volume, inclination, position_angle,
               mode="surface", threshold=0.01, project_vcam=False, vcam_derot_ang=0):
        """
        Render any scalar 3D volume (temperature OR gravity magnitude)
        to a 2D image after applying inclination + position angle.
        
        For radial velocity (which can be negative), set threshold to a small negative number
        or use mode='mip' (maximum intensity projection) which handles negatives better.

        Parameters:
        -----------
        volume : array-like
            3D array defined on the model's (x, y, z) grid. This can be the 
            volume returned by either build_temperature_volume() or 
            build_gravity_volume(), for example.
        inclination : float
            Inclination angle in degrees (0 = pole-on, 90 = edge-on)
        position_angle : float
            Position angle in degrees (0 = north up, positive eastward)
        mode : str, default "surface"
            Rendering mode: "surface" (first hit), "mip" (max intensity),
            or "volume" (accumulated with transparency)
        threshold : float, default 0.01
            Threshold for "surface" mode (minimum value to count as a hit)
            
        Returns:
        --------
        image : 2D array
            Rendered image of the volume after rotation
        
        """
        if self.x is None:
            raise RuntimeError("Build a volume first using build_temperature_volume(), "
                             "build_gravity_volume(), or build_radial_velocity_volume().")

        # Same rotation matrix as before
        PA_rotation_matrix = np.array([
            [np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
            [np.sin(np.radians(position_angle)),  np.cos(np.radians(position_angle)), 0],
            [0, 0, 1]
        ])
        inclination_rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
            [0, np.sin(np.radians(inclination)),  np.cos(np.radians(inclination))]
        ])
        R = PA_rotation_matrix @ inclination_rotation_matrix

        if project_vcam:
            R = get_vcam_projection_matrix(vcam_derot_ang) @ R
        
        # Auto-adjust threshold for volumes that can be negative (like radial velocity)
        if threshold is None:
            if np.nanmin(volume) < 0:
                threshold = np.nanmin(volume) * 0.1   # small negative number
            else:
                threshold = 0.01

        # Auto-detect signed data
        signed = np.nanmin(volume) < 0

        image = render_object(self.x, self.y, self.z, volume, R,
                              resolution=self.render_resolution,
                              mode=mode,
                              threshold=threshold,
                              signed=signed)  

        image[image == 0] = np.nan

        return image

    
    def get_pixel_scale(self, inclination, position_angle, project_vcam=False, vcam_derot_ang=0):
        """
        Compute the true on-sky pixel scale (meters per pixel) for a given orientation.
        
        This matches exactly what render_object() uses for the u/v bounding box.
        Returns scale in meters per pixel.
        """
        if self.x is None:
            self._ensure_grid()

        # Build the same rotation matrix as in render()
        PA_rotation_matrix = np.array([
            [np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
            [np.sin(np.radians(position_angle)),  np.cos(np.radians(position_angle)), 0],
            [0, 0, 1]
        ])
        inclination_rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
            [0, np.sin(np.radians(inclination)),  np.cos(np.radians(inclination))]
        ])
        R = PA_rotation_matrix @ inclination_rotation_matrix

        if project_vcam:
            R = get_vcam_projection_matrix(vcam_derot_ang) @ R

        # Project the 8 corners of the grid exactly as render_object does
        corners = np.array([
            [xi, yi, zi]
            for xi in (self.x[0], self.x[-1])
            for yi in (self.y[0], self.y[-1])
            for zi in (self.z[0], self.z[-1])
        ])  # shape (8, 3)

        corners_cam = corners @ R.T

        u_span = corners_cam[:, 0].max() - corners_cam[:, 0].min()
        v_span = corners_cam[:, 1].max() - corners_cam[:, 1].min()

        # The renderer uses the larger of the two spans + 1.02 margin, and makes a square image
        max_span = max(u_span, v_span) * 1.02

        pixel_scale = max_span / self.render_resolution   # meters per pixel

        return pixel_scale
    
    # ------------------------------------------------------------------
    # Surface projection utilities
    # ------------------------------------------------------------------
    @staticmethod
    def project_grid_on_sky(theta, phi, R, inclination, position_angle, color=None):
        """
        Project surface points onto the sky plane. This can be useful for 
        visualizing the model grid. 
        
        example usage:
        x, y, z, c = model.project_grid_on_sky(
            model.theta_grid.flatten(),
            model.phi_grid.flatten(),
            model.R_surface.flatten(),
            inclination, position_angle,
            color=model.theta_grid.flatten()
        )
        
        Parameters:
        -----------
        theta : array-like
            Colatitude of surface points (radians)
        phi : array-like
            Azimuthal angle of surface points (radians)
        R : array-like
            Radius of surface points (m)
        inclination : float
            Inclination angle in degrees (0 = pole-on, 90 = edge-on)
        position_angle : float
            Position angle in degrees (0 = north up, positive eastward)
        color : array-like, optional
            Optional array of values to return for each point (e.g., temperature)
            
        Returns:
        --------
        x_proj : array-like
            Projected x coordinates on the sky plane (m)
        y_proj : array-like
            Projected y coordinates on the sky plane (m)
        z_proj : array-like
            Line-of-sight coordinate (m, positive towards observer)
        color_proj : array-like
            Projected color values (same shape as input color, if provided)
        
        """
        PA_rot = np.array([
            [np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle)), 0],
            [np.sin(np.radians(position_angle)),  np.cos(np.radians(position_angle)), 0],
            [0, 0, 1]
        ])
        inc_rot = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(inclination)), -np.sin(np.radians(inclination))],
            [0, np.sin(np.radians(inclination)),  np.cos(np.radians(inclination))]
        ])
        proj_matrix = PA_rot @ inc_rot

        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)

        coords = np.stack([x.flatten(), y.flatten(), z.flatten()])
        projected = proj_matrix @ coords

        x_proj = projected[0].reshape(theta.shape)
        y_proj = projected[1].reshape(theta.shape)
        z_proj = projected[2].reshape(theta.shape)

        visible = z_proj > 0
        color = theta if color is None else color
        return (x_proj[visible], y_proj[visible], z_proj[visible],
                color[visible].flatten())




def render_object(x, y, z, c, R, resolution=256, mode="surface",
                  threshold=0.01, n_samples=None, signed=False):
    """
    Render a 3D greyscale volume as a 2D image using ray casting.
    This is to go from model volumes (array of points on a sphere) to an 
    actual image as seen by an observer after applying inclination and position
    angle.
    
    Parameters:
    -----------
    x, y, z : 1D arrays
        Coordinates of the 3D grid points (must match shape of c)
    c : 3D array
        Scalar values defined on the (x, y, z) grid (e.g., temperature or gravity)
    R : 3x3 array
        Rotation matrix to apply to the coordinates (from inclination and PA)
    resolution : int, default 256
        Resolution of the output image (resolution x resolution)
    mode : str, default "surface"
        Rendering mode: "surface" (first hit), "mip" (max intensity),
        or "volume" (accumulated with transparency)
    threshold : float, default 0.01
        Threshold for "surface" mode (minimum value to count as a hit)
    n_samples : int, optional
        Number of samples along each ray (only for "mip" and "volume" modes)
        
    Returns:
    --------
    image : 2D array
        Rendered image of the volume after rotation
    """
    R = np.asarray(R, dtype=float)
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if c.shape != (len(x), len(y), len(z)):
        raise ValueError(f"c.shape {c.shape} does not match grid dimensions "
                        f"({len(x)}, {len(y)}, {len(z)})")

    interp = RegularGridInterpolator(
        (x, y, z), c,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    cam_x = R[0]
    cam_y = R[1]
    cam_z = R[2]

    corners = np.array([
        [xi, yi, zi]
        for xi in (x[0], x[-1])
        for yi in (y[0], y[-1])
        for zi in (z[0], z[-1])
    ])
    corners_cam = corners @ R.T
    u_min, u_max = corners_cam[:, 0].min(), corners_cam[:, 0].max()
    v_min, v_max = corners_cam[:, 1].min(), corners_cam[:, 1].max()
    t_min, t_max = corners_cam[:, 2].min(), corners_cam[:, 2].max()

    u_ctr = (u_min + u_max) / 2
    v_ctr = (v_min + v_max) / 2
    half = max(u_max - u_min, v_max - v_min) / 2 * 1.02
    u_vals = np.linspace(u_ctr - half, u_ctr + half, resolution)
    v_vals = np.linspace(v_ctr - half, v_ctr + half, resolution)

    if n_samples is None:
        n_samples = int(3 * max(len(x), len(y), len(z)))
    t_vals = np.linspace(t_min, t_max, n_samples)

    UU, VV = np.meshgrid(u_vals, v_vals)
    origin = (UU[..., np.newaxis] * cam_x + VV[..., np.newaxis] * cam_y)

    image = np.zeros((resolution, resolution), dtype=float)

    if mode == "surface":
        hit_mask = np.zeros((resolution, resolution), dtype=bool)
        hit_color = np.zeros((resolution, resolution), dtype=float)

        # For signed data, use absolute value for "hit" detection
        hit_threshold = abs(threshold) if signed else threshold

        for t in t_vals:
            if hit_mask.all():
                break
            pts = origin + t * cam_z
            vals = interp(pts)
            
            if signed:
                new_hit = ~hit_mask & (np.abs(vals) > hit_threshold)
                hit_color[new_hit] = vals[new_hit]   # keep original sign
            else:
                new_hit = ~hit_mask & (vals > hit_threshold)
                hit_color[new_hit] = vals[new_hit]
            
            hit_mask |= new_hit

        image = hit_color

    elif mode == "mip":
        for t in t_vals:
            pts = origin + t * cam_z
            vals = interp(pts)
            np.maximum(image, vals, out=image)   # mip still works with negatives

    elif mode == "volume":
        dt = (t_max - t_min) / n_samples
        transmittance = np.ones((resolution, resolution), dtype=float)
        for t in t_vals:
            pts = origin + t * cam_z
            sigma = interp(pts)
            alpha = 1.0 - np.exp(-np.abs(sigma) * dt)   # absorption based on |sigma|
            image += transmittance * alpha * sigma      # but emission keeps sign
            transmittance *= 1.0 - alpha
    else:
        raise ValueError(f"Unknown mode {mode!r}.")

    return image



def get_vcam_projection_matrix(derot_ang):
    # Mirror across x-axis 
    mirror_x = np.array([[1.0,  0.0, 0.0],
                         [0.0, -1.0, 0.0], 
                         [0.0,  0.0, 1.0]])   
    
    angle = np.radians(derot_ang)
    derotate = np.array([[ np.cos(angle), -np.sin(angle), 0.0],
                         [ np.sin(angle),  np.cos(angle), 0.0],
                         [ 0.0,            0.0,           1.0]])
    
    # Something about the spectro-astrometry pipeline causes the image to be 
    # flipped in both x and y. I cannot find the source of this, so I regretfully 
    # patch it here. 
    mirror_xy = np.array([[-1.0,  0.0, 0.0],
                          [0.0, -1.0, 0.0], 
                          [0.0,  0.0, 1.0]])
    
    # Forward matrix for points: A = derotate @ mirror_x
    A = derotate @ mirror_x @ mirror_xy  
    
    # Compute inverse for affine_transform
    A_inv = np.linalg.inv(A)
    return A_inv



# ----------------------------------------------------------------------
# Example usage -- Lets generate an on-sky image
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # # STELLAR PARAMETERS (Altair) ---------------------------------------------
    M_star = 1.791 * M_sun.value    # stellar mass in kg
    R_p = 1.634 * R_sun.value       # polar radius of the star in meters
    T_eff = 7550.0                  # K, effective temperature (used only for normalizing C_omega)
    omega_frac = 0.923              # Fraction of critical rotation
    # inclination = 57.2              # degrees, 0 = pole-on, 90 = edge-on
    position_angle = -62.7          # degrees, 0 = north up, positive eastward
    beta = 0.25                     # UNTESTED FOR VALUES != 0.25; gravity darkening exponent (von Zeipel)
    project_vcam = True             # Whether to project on-sky or project on vcam (False for on-sky, True for vcam)
    derot_ang = 101.34              # degrees, angle to rotate the vcam projection to match on-sky orientation (only used if project_vcam=True)
    rigid = False                    # Whether to treat the star as a rigid rotator (no gravity darkening)    
    # -------------------------------------------------------------------------
    
    # inclination = 61.73
    # rigid = True
    
    # MODEL RESOLUTION --------------------------------------------------------
    N_grid = 2**5                   # 3D grid resolution (N x N x N)
    render_resolution = 2**5        # Final rendered image resolution
    grid_overfill = 1.1            # How much larger the grid should be than the star's polar radius (in units of R_p).
    # -------------------------------------------------------------------------
    
    results_dir = './renders'
    plots_dir = './plots'
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    
    
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
    
    # -------------------------------------------------------------------------
    
    # # %%
    
    # # Make some plots ---------------------------------------------------------
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # plt.figure(figsize=(8, 2.5), tight_layout=True)
    # plt.suptitle("Altair Rendered Images", fontsize=14)
    
    # plt.subplot(1, 3, 1)
    # plt.title('Temperature')
    # plt.imshow(image_T, origin='lower', cmap='plasma')
    # # plt.xlim(image_T.shape[1]//2 - image_T.shape[1]//5, image_T.shape[1]//2 + image_T.shape[1]//5)
    # # plt.ylim(image_T.shape[1]//2 - image_T.shape[1]//5, image_T.shape[1]//2 + image_T.shape[1]//5)
    # plt.colorbar(label='T [K]', fraction=0.046, pad=0.04)
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.title('Gravity')
    # plt.imshow(image_g, origin='lower', cmap='cividis')
    # # plt.xlim(image_g.shape[1]//2 - image_g.shape[1]//5, image_g.shape[1]//2 + image_g.shape[1]//5)
    # # plt.ylim(image_g.shape[1]//2 - image_g.shape[1]//5, image_g.shape[1]//2 + image_g.shape[1]//5)
    # plt.colorbar(label=r'$log_{10}(|g|)$ [cm s$^{-2}$]', fraction=0.046, pad=0.04)
    # plt.axis('off')
    
    # plt.subplot(1, 3, 3)
    # plt.title('Radial Velocity')
    # plt.imshow(image_vrad, origin='lower', cmap='RdBu_r')
    # # plt.xlim(image_vrad.shape[1]//2 - image_vrad.shape[1]//5, image_vrad.shape[1]//2 + image_vrad.shape[1]//5)
    # # plt.ylim(image_vrad.shape[1]//2 - image_vrad.shape[1]//5, image_vrad.shape[1]//2 + image_vrad.shape[1]//5)
    # plt.colorbar(label=r'$v_{rad}$ [km s$^{-1}$]', fraction=0.046, pad=0.04)
    # plt.axis('off')
    
    
    # prejection = 'no_proj' if not project_vcam else 'vcam_proj'
    # plt.savefig(os.path.join(plots_dir, f'rendered_star_{prejection}.png'), dpi=300)
    
    # # %%
    
    # # # project on sky usage (shows the grid points on the star's surface colored by colatitude)
    # # # useful for debugging the grid and understanding the geometry, but not necessary for the final result
    # # x,y,z,c = model.project_grid_on_sky(
    # #     model.theta_grid.flatten(),
    # #     model.phi_grid.flatten(),
    # #     model.R_surface.flatten(),
    # #     inclination, position_angle,
    # #     # color=v_rad_vol.flatten()
    # #     color=model.theta_grid.flatten()
    # # )
    
    # # plt.figure(figsize=(6, 6))
    # # plt.scatter(x, y, c=c, s=0.1, cmap='viridis', marker='.')
    # # plt.colorbar(label='Colatitude (rad)')
    # # plt.title('Projected Stellar Surface')
    # # plt.xlabel('x (m)')
    # # plt.ylabel('y (m)')
    # # plt.axis('equal')
    # # plt.grid()
    # # plt.savefig(os.path.join(plots_dir, 'projected_stellar_surface.png'), dpi=300)
    
        
        
    # print('Making a 3D plot')
    # # UNCOMMENT FOR 3D HTML PLOT ----------------------------------------------
    # # NOTE: This can only be run for high resolution grids (N>=2**8)
    # if N_grid <= 2**8:
    #     import sys
    #     sys.exit(0) 
    # import plotly.graph_objects as go
    # from skimage import measure

    # # Your data
    # x, y, z = model.x, model.y, model.z
    # values = np.asarray(T_vol, dtype=float)

    # # Handle NaNs (replace with a value outside the range of interest, e.g. very low temp)
    # values = np.nan_to_num(values, nan=np.nanmin(values) - 100)

    # # Create 3D grid
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # # === Extract the outer isosurface using marching cubes ===
    # # Choose an isovalue near the "surface" of the star.
    # # Good starting point: something like 0.5 * max(T) or experiment with percentiles
    # isovalue = np.nanpercentile(values, 70)   # adjust this (higher = smaller surface)

    # verts, faces, normals, values_on_surface = measure.marching_cubes(
    #     values, 
    #     level=isovalue,
    #     spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]) if len(x)>1 else (1,1,1)
    # )
    # import plotly.graph_objects as go
    # from skimage import measure

    # # Your data
    # x, y, z = model.x, model.y, model.z
    # values = np.asarray(T_vol, dtype=float)

    # # Handle NaNs (replace with a value outside the range of interest, e.g. very low temp)
    # values = np.nan_to_num(values, nan=np.nanmin(values) - 100)

    # # Create 3D grid
    # X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # # verts now contains (x,y,z) coordinates of the surface points
    # # values_on_surface contains the original temperature values interpolated on the surface vertices

    # fig = go.Figure(data=go.Mesh3d(
    #     x=verts[:, 0],
    #     y=verts[:, 1],
    #     z=verts[:, 2],
    #     i=faces[:, 0],
    #     j=faces[:, 1],
    #     k=faces[:, 2],
        
    #     # Color the surface using temperature values on the vertices
    #     intensity=values_on_surface,          # this drives the coloring
    #     colorscale='Viridis',                 # try 'Plasma', 'Inferno', 'Hot', 'RdYlBu'
    #     showscale=True,                       # shows the colorbar
    #     colorbar=dict(title='Temperature'),
        
    #     # Lighting & appearance tweaks for a nice 3D look
    #     flatshading=False,                    # smoother shading
    #     lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.5),
    #     lightposition=dict(x=100, y=100, z=100),
        
    #     # Opacity (1.0 = fully opaque surface)
    #     opacity=1.0
    # ))

    # fig.update_layout(
    #     title='Outer Surface Temperature of Rotating Star',
    #     scene=dict(
    #         xaxis_title='X (m)',
    #         yaxis_title='Y (m)',
    #         zaxis_title='Z (m)',
    #         aspectmode='data',                # preserves physical proportions
    #         camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    #     ),
    #     width=1000,
    #     height=800
    # )

    # fig.write_html('star_surface_temperature.html')
    # # fig.show()  # uncomment to view interactively

    # # verts now contains (x,y,z) coordinates of the surface points
    # # values_on_surface contains the original temperature values interpolated on the surface vertices

    # fig = go.Figure(data=go.Mesh3d(
    #     x=verts[:, 0],
    #     y=verts[:, 1],
    #     z=verts[:, 2],
    #     i=faces[:, 0],
    #     j=faces[:, 1],
    #     k=faces[:, 2],
        
    #     # Color the surface using temperature values on the vertices
    #     intensity=values_on_surface,          # this drives the coloring
    #     colorscale='Viridis',                 # try 'Plasma', 'Inferno', 'Hot', 'RdYlBu'
    #     showscale=True,                       # shows the colorbar
    #     colorbar=dict(title='Temperature'),
        
    #     # Lighting & appearance tweaks for a nice 3D look
    #     flatshading=False,                    # smoother shading
    #     lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.5),
    #     lightposition=dict(x=100, y=100, z=100),
        
    #     # Opacity (1.0 = fully opaque surface)
    #     opacity=1.0
    # ))

    # fig.update_layout(
    #     title='Outer Surface Temperature of Rotating Star',
    #     scene=dict(
    #         xaxis_title='X (m)',
    #         yaxis_title='Y (m)',
    #         zaxis_title='Z (m)',
    #         aspectmode='data',                # preserves physical proportions
    #         camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    #     ),
    #     width=1000,
    #     height=800
    # )

    # fig.write_html(os.path.join(plots_dir, 'star_surface_temperature.html'))
    # # fig.show()  # uncomment to view interactively
        
        
        
        
        
        