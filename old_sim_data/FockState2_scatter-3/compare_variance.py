import numpy as np
import matplotlib.pyplot as plt

# %% define constants
r0 = [2]
kappa = [0, 1.5, 3]

t_max = 75

path = 'FockState2_scatter-3'
# %% load data
example_dataset = np.load(f'{path}/model_r_1_kap0_chris.npy')[0]
model_chris = np.empty([len(r0), len(kappa), len(example_dataset)])
model_matteo = np.empty([len(r0), len(kappa), len(example_dataset)])
mps_chris = np.empty([len(r0), len(kappa), len(example_dataset)])

for ir, r in enumerate(r0):
    for ik, k in enumerate(kappa):
        model_chris[ir, ik] = np.load(f'{path}/model_r_{r}_kap{k}_chris.npy')[0]
        model_matteo[ir, ik] = np.load(f'{path}/k_{k}_{r}.npy') - (r ** 2 - 1) / 12
        mps_chris[ir, ik] = np.load(f'{path}/mps_r_{r}_kap{k}_chris.npy')[0]
ts = np.linspace(0, 10, 101)

# cleanup zeros
model_chris[model_chris < 1e-20] = 0
model_matteo[model_matteo < 1e-20] = 0
mps_chris[mps_chris < 1e-20] = 0

# %% absolute comparison of mu_2
r_fix = 16
plt.subplots(1, 1, figsize=(3, 2.5), dpi=400)
for ir, r in enumerate(r0):
    for ik, k in enumerate(kappa):
        if r != r_fix:
            continue
        # plt.plot(ts[:t_max], model_chris[ir, ik, :t_max], '.', label=rf'mod chris $\kappa={k},r_0={r}$')
        plt.scatter(ts[:t_max], model_matteo[ir, ik, :t_max], s=30, marker='d', label=rf'$\kappa={k}$',
                    c=f'C{ik}')
        plt.scatter(ts[:t_max], mps_chris[ir, ik, :t_max], s=20, marker='.', color=f'C{ik}', edgecolor='black')
        # , label=rf'$\kappa={k},r_0={r}$',color=f'C{ik}')
plt.yscale('log')
plt.xscale('log')
plt.ylim([0.001, 150])
plt.title(f'Variance $\mu_2, r_0={r_fix}$')
plt.legend(loc='lower right', ncol=1, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig(f'{path}/variance_r{r_fix}.pdf')
plt.show()

# %% absolute deviation from model
plt.subplots(1, 1, figsize=(3, 2.5), dpi=400)
for ir, r in enumerate(r0):
    for ik, k in enumerate(kappa):
        if r != r_fix:
            #plt.plot(ts[:t_max], np.abs(model_matteo - model_chris)[ir, ik,:t_max], 'x')#, label='')
            plt.plot(ts[:t_max], np.abs(model_matteo - mps_chris)[ir, ik,:t_max]/np.abs(model_matteo + mps_chris)[ir, ik,:t_max], '.')#, label='mps')
plt.yscale('linear')
plt.xscale('linear')
plt.title('absolute deviation\nmy model\nS matteos model')
#plt.legend(loc='right')
plt.tight_layout()
plt.show()

# %% relative deviation from model
plt.subplots(1, 1, figsize=(3, 2.5), dpi=400)
for ir, r in enumerate(r0):
    for ik, k in enumerate(kappa):
        if r != r_fix:
            #plt.plot(ts[:t_max], np.abs(model_matteo - model_chris)[ir, ik,:t_max], 'x')#, label='')
            plt.plot(ts[:t_max], np.abs(model_matteo - mps_chris)[ir, ik,:t_max]/np.abs(model_matteo + mps_chris)[ir, ik,:t_max], '.')#, label='mps')
plt.yscale('logS')
plt.xscale('linear')
plt.title('relative deviation\nMPS vs model')
#plt.legend(loc='right')
plt.tight_layout()
plt.show()
