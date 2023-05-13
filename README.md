# Tweezer-Dynamics
[![DOI](https://zenodo.org/badge/640364467.svg)](https://zenodo.org/badge/latestdoi/640364467)
**A cluster optimized Python based simulation framework to simulate Rydberg tweezer of 1D dynamics utilizing tensor network methods.**


The parameters of the simulation can be changed in `mps_simulation.py`. Different initial states and hamiltonians can be added in the file `tweezer_dynamics_utils.py`.

During the simulation, after every timestep all results are stored to a new dataframe in HDF5 file format.
This leads to large files with redundant values while adding a layer of security for timeouts or crashing simulations.
Moreover, a SLURM array job launched with `runArrayJob.slurm` creates many of those files.
To remove the redundancy and combine array result data after a successful simulation ,the script `minify_h5_results.py` can be used.

Data can be plotted and analyzed with `plot_dataset.py`.

Old simulation data with corresponding figures can be found in `./data/`

The following features are supported already:
- [x] Supply a list of $\Omega$-values to simulate
- [x] Supply a list of $V$-values to simulate
- [x] Specify the number of atoms $N$
- [x] Specify a list of single- and multi-site observables you want to measure the time-evolution during the simulation
- [x] Supply a list of time-points you want to calculate the observables
- [x] Specify numeric tolerance and timeout thresholds
- [x] Specify trap-frequencies $\omega_\mathrm{trap}$ which model delocalization of atoms in a tweezer-trap
- [ ] Calculate $V$ directly via interatomic distance $a_0$ and given coefficients $C_3$ and $C_6$
- [ ] correct formula is then: $V(r_j,r_k)=V(a_0)+\sum_\nu^{\inf} d/dr^\nu V(r)|_{a_0} (a_j^\dagger+a_j-a_k^\dagger - a_k)$
- [ ]  Check whether higher order potential expansions are worth
- [ ] color{blue}{They are, when $\omega$ is small.}\textcolor{red}{CHECK: When the spreading $\Delta x$ in the harmonic traps is larger than $a_0$.}
- [ ]  Set atom at one harmonic oscillation level corresponding to a specific temperature $(n=1,2,3...)$
- [x]  Specify a distribution of levels $n$ for the atoms. For example $80\%$ in $n=1$, $20\%$ in $n=2$.
- [ ]  Add long-range interaction beyond nearest-neighbour interactions
- [ ]  Specify the order of long-range interaction
- [ ]  Supply a µ-wave tuned interaction potential (maybe a potential of this form can also be written as $C_3/r^3+C_6/r^6$, so the simulation is already able to handle it)
- [ ] Do simulation with realistic potential with the $C_3$ and $C_6$ values corresponding to a specific pair-state. We have to change the Hamiltonian for this. We combine now the phonons and spins to a local quantum object of dimension $2M$. In this fashion we can extend the mps-simulation with phonon-modes.
- [ ]  Estimate errors of observables
- [ ]  Add dissipative processes like radiative decay, dephasing etc.
- [ ]  Specify different potentials for trapped and anti-trapped Rydberg atoms. Excited Rydberg atoms are less or even anti-trapped compared to well trapped ground-states. Not relevant, if traps are off. Maybe do Quench dynamics?
- [ ] Do simulation with exact calculated pair-potential out of i.e. 