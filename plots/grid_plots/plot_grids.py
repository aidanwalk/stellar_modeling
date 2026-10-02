import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')



# Import stellar modeling modules from two directories up in the hierarchy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import Stellar3DModel



if __name__ == "__main__":
    from astropy.constants import M_sun, R_sun
    
    M_star = 2*M_sun.value
    R_star = 2*R_sun.value
    T_star = 6000  # Example temperature in Kelvin
    v_rot = 100e3  # Example rotational velocity in m/s
    PA = 65.0
    inc = 35
    
    
    model = Stellar3DModel.Stellar3DModel(
        M_star, 
        R_star,
        T_star, 
        v_rot = v_rot, 
        N_grid=2**7,
        render_resolution=2**9,
        rigid=False, 
        overfill=1.2
        )
    
    T_vol = model.build_temperature_volume()
    image_T = model.render(T_vol, position_angle=PA, inclination=inc)
    
    
    plt.figure(tight_layout=True)
    plt.imshow(image_T, origin='lower', cmap='viridis')
    plt.colorbar(label="Temperature (K)")
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rendered Stellar Temperature Map")
    plt.savefig("rendered_stellar_temperature_map.png")