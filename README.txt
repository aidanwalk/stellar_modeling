Generate simulated data to feed into the spectro-astrometry pipeline:


1. Generate an on-sky image of Altair, where intensity values in the image
   represent physical parameters on that point of Altair's surface. We want 
   parameters such as temperature, gravity, and velocity, which largely 
   determine the emitted spectrum at that point. 
   
   This is done in `Stellar3DModel.py`, which constructs a 3D model of Altair's
   surface and renders it into a 2D image. To generate these images, run:
   
   ```
   python Stellar3DModel.py
   ```

2. Construct a spectral image of Altair, where each pixel contains the spectrum 
   emitted from that point on the star's surface. This is done in 
   'StellarSpectralImage.py', which takes the rendered images from step 1 and 
   uses them to query spectal templates from the PHOENIX library. 
   
   To generate the spectral image, run:
   
   ```
   python StellarSpectralImage.py
   ```


3.0 Deshift the on-sky maps to remove the signal in the data. De-shifts are 
   calculated by first fitting the optimal shift between the on-sky maps to
   the reference maps, then de-shifting the on-sky maps by that amount. This is
   done in `DeshiftMaps.py`. The result is a set of noisey observed maps that 
   contain no signal. To generate the de-shifted maps, run:
   
   ```
   python DeshiftMaps.py
   ```


3. Convolve the spectral image with the observed reference maps to generate
   simulated observed maps. This is done in `ConvolveMaps.py`, which takes the 
   spectral image from step 2 and convolves it with the observed reference maps
   to generate simulated observed maps. 
   
   To generate the simulated observed maps, run:
   
   ```
   python ConvolveMaps.py
   ```



4. Measure the photocenter shifts in the simulated observed maps using the same 
   spectro-astrometry pipeline used on the real data. This is done in 
   `CentroidShifts.py`, which takes the simulated observed maps from step 3 and
   runs them through the spectro-astrometry pipeline to measure the photocenter
   shifts. 
   
   To measure the photocenter shifts, run:
   
   ```
   python CentroidShifts.py
   ```