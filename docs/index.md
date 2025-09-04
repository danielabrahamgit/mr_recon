# Home

## Overview
**mr_recon** is a pytorch library for **M**agnetic **R**esonance **Recon**struction techniques. It is heavily inspired by the [sigpy](https://sigpy.readthedocs.io/en/latest/index.html) library, with some changes:   
- Pytorch instead of cupy for GPU computing.   
- Added more MR realted functionality like Gfactor computation, GRAPPA, rovir, etc.  
- More emphasis on efficient NUFFT and non-Cartesian approaches.   
- Extensive spatio-temporal phase modeling techniques (useful for field imperfections).  


## Installation
To install, first clone the repository. Then create a conda environment
```
conda env create -n mr_recon --file environment.yml
```

Activate the environment:
```
conda activate mr_recon
```

and finally install mr_recon in debug mode 
```
pip install -e ./
```


## Tutorials
See the [demo page](examples/sense.ipynb)