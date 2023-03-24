# Solver for acoustic scattering problem

This repository contains the code generating the numerical results in Chapter 5 of the thesis "Sensitivity-Guided Shape Reconstruction".

## Dependencies

The implementation depends on the following Python packages:

- [Firedrake](https://www.firedrakeproject.org/documentation.html)

- [Gmsh](https://gmsh.info/)

- NumPy

- Matplotlib

## File description

- `mesh.py`

    The function `generate_mesh` uses [Gmsh](https://gmsh.info/) to create mesh that is compatible with the PML method and far field pattern evaluation.

- `scattering.py`

    This module implements solver of acoustic scattering problem using PML method and far field evaluation using boundary- and volume-based formulas. It also includes functions for various visualization purposes.

- `point_source.ipynb`

    This notebook validates the implementation of the PML method.

- `plane_wave.ipynb`

    This notebook validates the implementation of the far field evaluation.

## Usage

1. Install the finite element library [Fireshape](https://www.firedrakeproject.org/documentation.html).

        curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
        python3 firedrake-install

2. Activate Firedrake's virtualenv.

        source path-to-firedrake/bin/activate

3. Install the finite element mesh generator [Gmsh](https://gmsh.info/).

        pip install --upgrade gmsh

4. Add Firedrake's virtualenv to Jupyter Notebook.

        pip install --user ipykernel
        python3 -m ipykernel install --user --name=firedrake

5. Open Jupyter Notebook and select kernel `firedrake`.

## References

- Bermúdez, A., Hervella-Nieto, L., Prieto, A., Rodríguez, R., 2007. An optimal perfectly matched layer with unbounded absorbing function for time-harmonic acoustic scattering problems. Journal of Computational Physics 223, 469–488. https://doi.org/10.1016/j.jcp.2006.09.018

- Colton, D., Kress, R., 2019. Inverse Acoustic and Electromagnetic Scattering Theory, Applied Mathematical Sciences. Springer International Publishing, Cham. https://doi.org/10.1007/978-3-030-30351-8
