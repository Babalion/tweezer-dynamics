import multiprocessing
import sys
import uuid
from datetime import datetime
from functools import partial
from time import time

import h5py
import quimb as qu
from tqdm import tqdm

from tweezer_dynamics_utils import *

# %% ==================================================
# choose parameters
# =====================================================
# simulation parameters
CUTOFF_TOLERANCE = 1e-7
TROTTER_TOLERANCE = 1e-4
MAX_BOND = 200
TIMEOUT_LIMIT_S = 1e5  # if a simulation step takes longer than this, we proceed with the next one
STORE_STATE = False  # weather we store the current MPS to a file or not
filename = f'Exact_vs_MPS/Exact_vs_MPS_j002'
print(f'Simulation gets stored to: {filename}')
filenameAlternative = filename + '2'
is_cyclic = True
dtype = 'complex128'

# physical parameters
num_spins = 7
dim_phonons = 3  # size of the boson Hilbert space
ryd_size = 1
spin_dim = 2

iterate_parameters = {
    'omega': np.linspace(8, 14, 1),
    'OmegaE': np.linspace(1, 5, 1),
    'OmegaR': np.linspace(1, 15, 1),
    'DeltaEE': [-200],
    'DeltaRR': [-200],
    'omegaR': [0],
    'V_e_vdw': [200],
    'V_r_vdw': [200],
    'V_dd': [0],
    'kappa_e': np.linspace(3, 3, 1),
    'kappa_r': np.linspace(0, 3, 1),
    'kappa_dd': np.linspace(0, 3, 1),
    'alpha': np.linspace(0.0, 3.14, 1),
    'beta': np.linspace(0.01, 10, 1),
}

# times we are interested in
ts = np.linspace(0, 3, 50)  # in µs

# %% SLURM job array initialization
# we can create a SLURM array job when supplying two additional parameters to the script
SLURM_ARRAY_JOB = False
jobId = 0
jobTotalNum = 1
# TODO utilize python argparse
# TODO add support for automatic slurm `sbatch` creation and submission
# TODO refactor array and single sbatch mode
try:
    if len(sys.argv) > 3:
        raise SyntaxWarning()
    jobId = int(sys.argv[1])
    jobTotalNum = int(sys.argv[2])
    print(f'started in **SLURM array mode** with id {jobId} of total {jobTotalNum}')
    SLURM_ARRAY_JOB = True
    filename += f'_j{jobId:03d}'
    filenameAlternative += f'_j{jobId:03d}'
except IndexError:
    print('started in **normal mode**')
except SyntaxWarning:
    print(f'WARNING: more than two parameters were given')
    print('started in **normal mode**')

# %% ==================================================
# DEFINE OBSERVABLES
# =====================================================
observables = {
    'single_site': {
        'e': qt.tensor(qt.basis(spin_dim, 1) * qt.basis(spin_dim, 1).dag(), qt.qeye(dim_phonons)).data.todense(),
        'n_a': qt.tensor(qt.qeye(spin_dim), qt.num(dim_phonons)).data.todense(),
    },
    'other': ['bond_dim', 'entropy']
}

# measure the whole phonon basis
for n in range(dim_phonons):
    observables['single_site'][f'phonon_{n}'] = qt.tensor(qt.qeye(spin_dim), qt.projection(dim_phonons, n, n))

# measure the 3rd level occupation in a 3-level spin
if spin_dim == 3:
    observables['single_site']['r'] = qt.tensor(qt.basis(spin_dim, 2) * qt.basis(spin_dim, 2).dag(),
                                                qt.qeye(dim_phonons)).data.todense()

# %% ==================================================
# INITIAL STATE ARRAY
# =====================================================
fock_state_array = 0 * np.ones([num_spins], dtype=int)
# fock_state_array[num_spins // 2 - 4] = 2
# fock_state_array[num_spins // 2 + 6] = 4

listOfInitialStates = [
    ClusterFockState(spin_dim=spin_dim, fock_state_array=fock_state_array, cluster_size=ryd_size,
                     num_spins=num_spins,
                     dim_phonons=dim_phonons,
                     is_cyclic=is_cyclic),
    # ClusterCoherentAndersonState(cluster_size=ryd_size, num_spins=num_spins, dim_phonons=dim_phonons, is_cyclic=is_cyclic),
    # Cluster01State(cluster_size=ryd_size, num_spins=num_spins, dim_phonons=dim_phonons, is_cyclic=is_cyclic)
]
iterate_parameters['initialState'] = listOfInitialStates


# %% ==================================================
# construct Hamiltonian
# =====================================================
def hamiltonian_from_params(**kwargs):
    """
    Creates the Hamiltonian with given parameters.
    Decides between 2 and 3 level spins
    :return: LocalHam1D
    """
    if spin_dim == 2:
        return hamiltonian_2_level_spin(num_spins=num_spins, dim_phonons=dim_phonons, is_cyclic=is_cyclic, **kwargs)
    elif spin_dim == 3:
        return hamiltonian_3_level_vee(num_spins=num_spins, dim_phonons=dim_phonons, is_cyclic=is_cyclic,
                                       **kwargs)
    else:
        raise NotImplementedError('This spin dimension is not implemented.')


# %% =================================
# CALCULATION OF TIME EVOLUTION
# ====================================

tasks = split_in_chunks(dict_product(iterate_parameters), jobTotalNum)
print(f'divided tasks in {len(tasks)} chunks.')

# TODO transform parameters into a labeled list
for param_set in tqdm(tasks[jobId]):
    obs_flat = flatten([list(observables[s]) for s in list(observables)])
    res = -1 * np.ones([num_spins, len(obs_flat), len(ts)], dtype=dtype)
    param_set['initialState'].set_params(**param_set)
    tebd = qtn.TEBD(
        param_set['initialState'].get_state(),
        hamiltonian_from_params(**param_set)
    )
    tebd.split_opts['cutoff'] = CUTOFF_TOLERANCE
    tebd.split_opts['max_bond'] = MAX_BOND
    print("used time-step: " + str(tebd.choose_time_step(tol=TROTTER_TOLERANCE, T=ts[-1], order=4)))

    breakLoop = False
    tStart = time()
    # calculate the states psit for each timestep
    for iStep, psit in enumerate(tebd.at_times(ts, tol=TROTTER_TOLERANCE, order=4)):
        # TODO add error handling due to convergence problems
        #  raise ValueError("Internal algorithm failed to converge.")

        # calculate expectation values parallelized over sites
        func = partial(expect_one_site, psi=psit, observables=observables, num_spins=num_spins, dtype=dtype,
                       is_cyclic=is_cyclic)
        pool_obj = multiprocessing.Pool()
        totalRes = pool_obj.map(func, range(num_spins))
        res[:, :, iStep] = totalRes
        tNow = time()
        tStep = tNow - tStart
        tStart = tNow
        if tStep > TIMEOUT_LIMIT_S:  # when step took more than ...s, break and continue with next values
            print('Timed out. Skipped this values...')
            breakLoop = True
            break
        if not breakLoop:  # only jump in here when not break loop before
            # create a hdf5 file with write access to store the data
            try:
                f = h5py.File(filename + '.h5', 'a')
            except BlockingIOError:
                f = h5py.File(filenameAlternative + '.h5', 'a')
            if STORE_STATE:
                qu.save_to_disk(tebd, filename + f'{iStep}.dmp', compress=3)
            # store now all calculated values to a hdf5 dataset
            res_ds = f.create_dataset(uuid.uuid4().hex, data=res, compression="gzip", compression_opts=9)
            # TODO save observable data in separate dataframes
            res_ds.attrs['observables'] = obs_flat
            # store metadata which correspond to the dataset
            # TODO: refactor by stepping through iterate_parameters
            res_ds.attrs['N_spins'] = num_spins
            res_ds.attrs['spin_dim'] = spin_dim
            res_ds.attrs['N_phonons'] = dim_phonons
            res_ds.attrs['max_bond'] = tebd.split_opts['max_bond']
            res_ds.attrs['Omega'] = param_set['OmegaE']  # just for backwards compatability
            res_ds.attrs['OmegaE'] = param_set['OmegaE']
            res_ds.attrs['OmegaR'] = param_set['OmegaR']
            res_ds.attrs['omegaTrap'] = param_set['omega']
            res_ds.attrs['omega'] = param_set['omega']  # just for backwards compatability
            res_ds.attrs['V_vdw'] = param_set['V_e_vdw']  # just for backwards compatability
            res_ds.attrs['V_e_vdw'] = param_set['V_e_vdw']
            res_ds.attrs['V_r_vdw'] = param_set['V_r_vdw']
            res_ds.attrs['V_dd'] = param_set['V_dd']
            res_ds.attrs['DeltaEE'] = param_set['DeltaEE']
            res_ds.attrs['DeltaRR'] = param_set['DeltaRR']
            res_ds.attrs['omegaR'] = param_set['omegaR']
            res_ds.attrs['kappa'] = param_set['kappa_e']  # just for backwards compatability
            res_ds.attrs['kappa_e'] = param_set['kappa_e']
            res_ds.attrs['kappa_r'] = param_set['kappa_r']
            res_ds.attrs['kappa_dd'] = param_set['kappa_dd']
            res_ds.attrs['CUTOFF_TOLERANCE'] = CUTOFF_TOLERANCE
            res_ds.attrs['TROTTER_TOLERANCE'] = TROTTER_TOLERANCE
            res_ds.attrs['times'] = ts[:iStep + 1]
            res_ds.attrs['Initial state'] = param_set['initialState'].get_label()
            res_ds.attrs['alpha'] = param_set['alpha']
            res_ds.attrs['beta'] = param_set['beta']
            res_ds.attrs['Date'] = datetime.now().timestamp()
            # close the file
            f.close()
