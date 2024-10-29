# mphot
*mphot* is a Python package to model photometry data from the ground or space.

![example plots](examples/example-plots.png)


## How it works

Simply put, 
- it combines user submitted [telescope * filter * camera qe] efficiencies with generic stellar models and sky transmission/radiance models (for Paranal, 2400m) to generate integrable grids of stellar fluxes and sky radiances.

- Then, *mphot* uses the grids to interpolate between different
    - atmospheric parameters (PWV, airmass)
    - target star parameters (effective temperature + distance)

- using user submitted
    - telescope/site parameters (primary and secondary diameters, site seeing)
    - camera parameters (plate scale, dark current, read noise, well depth, target well fill, read time)
    
- to calculate the ideal exposure time and expected precision for a given observation.

Please see the [examples](https://github.com/ppp-one/mphot/tree/main/examples) for more details on how to use *mphot*. For further details on the models used, please see [https://doi.org/10.1117/12.3018320](https://doi.org/10.1117/12.3018320).

Note, it uses stellar parameters from "[A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)". Temperatures between 1278 K to 3042 K are calibrated for the [SPECULOOS target list](https://doi.org/10.1051/0004-6361/202038827) with [2MASS](https://irsa.ipac.caltech.edu/Missions/2mass.html) (see Figure 4.7 in "[Optimised ground-based near-infrared instrumentation for robotic exoplanet transit surveys](https://doi.org/10.17863/CAM.96904)").


## Installation

You can install *mphot* in a Python (`>=3.11`) environment with

```bash
pip install mphot
```

or from a local clone

```bash
git clone https://github.com/ppp-one/mphot
pip install -e mphot
```

You can test the package has been properly installed with

```bash
python -c "import mphot"
```
