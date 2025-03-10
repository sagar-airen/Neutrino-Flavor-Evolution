# Documentation for Neutrino Flavor Evolution Solver

## Overview
This Python script numerically solves the flavor evolution of neutrinos and anti-neutrinos emitted from infinite line sources. The system assumes that occupation numbers and fluxes are stationary and homogeneous. The underlying partial differential equation governing this setup is:

```math
 i(v_x \partial_x + v_z \partial_z)\rho_{E,\textbf{v}} = [H, \rho_{E,\textbf{v}}] 
```

where the Hamiltonian \( H \) includes mass, energy, interaction terms, and an integral over phase space as follows

```math
H = \frac{M^2}{2E} + \sqrt{2} G_F \left[ N_\ell + \int d\Gamma' \, (1 - \mathbf{v} \cdot \mathbf{v}') \, \rho_{E', \mathbf{v}'} \right]
```
($M^2$ is the neutrino mass matrix, $N_\ell$ provides the usual Wolfenstein matter effect and the last term is the self-interaction term).

Due to computational constraints, periodic boundary conditions are imposed instead of simulating an infinite source.

## Dependencies
The script requires the following Python libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `scipy.fftpack`

## Main Components
### Imports and Setup
- Imports necessary scientific computing libraries.
- Configures numerical settings for computation.

### Differential Equation Solving
- Uses `scipy.integrate.ode` and `complex_ode` to numerically integrate the evolution equations.
- Implements periodic boundary conditions.

### Fourier Transformations
- Uses `scipy.fftpack` to perform fast Fourier transforms (FFT) to handle spatial dependence.

### Visualization
- Utilizes `matplotlib` to generate plots illustrating neutrino evolution over space and time.

## Usage
Run the script in a Python environment with all dependencies installed. Ensure that the system parameters (e.g., mass, energy, interaction terms) are set appropriately for the desired simulation.

## Notes
- The script assumes a homogeneous and stationary system.
- The boundary conditions are periodic to approximate an infinite system.
- The code uses numerical integration techniques, which may require fine-tuning of step sizes for accuracy.

## Future Improvements
- Implement adaptive step-size integration for improved accuracy.
- Introduce additional boundary condition options beyond periodic constraints.

