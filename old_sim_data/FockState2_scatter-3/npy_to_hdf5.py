import uuid
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np

# %% load data
r0 = 2
kappa = 3

path_name = 'FockState2_scatter-3'
R = np.load(f'{path_name}/k_3_rydb.npy')
B = np.load(f'{path_name}/k_3_bos.npy')
print(R.shape)
R_h5 = np.zeros([9, 63, 41])
R_h5[:, 0, :] = R.transpose()
R_h5[:, 1, :] = B.transpose()

# %%

plt.pcolormesh(R_h5[:,0,:])
plt.show()

# %% save data

f = h5py.File(f'{path_name}/model_matteo_k{kappa}_j000.h5', 'w')
res_ds = f.create_dataset(uuid.uuid4().hex, data=R_h5, compression="gzip", compression_opts=9)
# TODO save observable data in separate dataframes
res_ds.attrs['observables'] = ['e','n_a',] + ['None'] * 61
# store metadata which correspond to the dataset
# TODO: refactor by stepping through iterate_parameters
res_ds.attrs['N_spins'] = 9
res_ds.attrs['spin_dim'] = 2
res_ds.attrs['N_phonons'] = 3
res_ds.attrs['max_bond'] = -1
res_ds.attrs['Omega'] = 1  # just for backwards compatability
res_ds.attrs['OmegaE'] = 1
res_ds.attrs['OmegaR'] = 0
res_ds.attrs['omegaTrap'] = 8
res_ds.attrs['omega'] = 8  # just for backwards compatability
res_ds.attrs['V_vdw'] = -1  # just for backwards compatability
res_ds.attrs['V_e_vdw'] = -1
res_ds.attrs['V_r_vdw'] = -1
res_ds.attrs['V_dd'] = -1
res_ds.attrs['DeltaEE'] = -1
res_ds.attrs['DeltaRR'] = -1
res_ds.attrs['omegaR'] = -1
res_ds.attrs['kappa'] = kappa  # just for backwards compatability
res_ds.attrs['kappa_e'] = kappa
res_ds.attrs['kappa_r'] = -1
res_ds.attrs['kappa_dd'] = -1
res_ds.attrs['CUTOFF_TOLERANCE'] = -1
res_ds.attrs['TROTTER_TOLERANCE'] = -1
res_ds.attrs['times'] = np.linspace(0, 4, 41)
res_ds.attrs['Initial state'] = f'{r0} initial centered excited Rydbergs, custom fock state'
res_ds.attrs['alpha'] = 0
res_ds.attrs['beta'] = 0
res_ds.attrs['Date'] = datetime.now().timestamp()
# close the file
f.close()
