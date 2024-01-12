import re

import h5py
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{physics}',
    "font.family": "serif",
    "font.serif": "STIX",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "font.weight": 'bold',
})
# %% load file and set path

# store the plots in the same folder as the data are
subfolder = 'FockState_r0_sample'

# specify if the initial state has an even or odd number of excited Rydbergs
even_state = True
isAlphaState = True
showPhononTrunc = False
showImage = True

# We can plot only datasets with equal times
f = h5py.File(f'./old_sim_data/{subfolder}/{subfolder}.h5', 'r')
# %% print list of kappa and omega which are in the dataset
datasetParams = pd.DataFrame(f.keys(), columns=['key'], dtype=str)
datasetParams['endTime'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['times'][-1], axis=1)
datasetParams['omegaTrap'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['omegaTrap'], axis=1)
datasetParams['DeltaEE'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['DeltaEE'], axis=1)
datasetParams['kappa'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['kappa'], axis=1)
datasetParams['cutoff'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['CUTOFF_TOLERANCE'], axis=1)
datasetParams['max_bond'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['max_bond'], axis=1)
datasetParams['trotter'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['TROTTER_TOLERANCE'], axis=1)
datasetParams['N_phonons'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['N_phonons'], axis=1)
datasetParams['N_spins'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['N_spins'], axis=1)
datasetParams['alpha'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['alpha'], axis=1)
datasetParams['beta'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['beta'], axis=1)
datasetParams['numObs'] = datasetParams.apply(lambda r: len(f.get(r['key']).attrs['observables']), axis=1)
datasetParams['Initial state'] = datasetParams.apply(
    lambda r: re.search(r'\d+', str(f.get(r['key']).attrs['Initial state'])),
    axis=1)
datasetParams['r0'] = datasetParams.apply(lambda r: int(np.round(np.sum(np.array(f.get(r['key'])).real[:, 0, 0]), 0)),
                                          axis=1)
datasetParams['Fock'] = datasetParams.apply(lambda r: int(np.array(f.get(r['key'])).real[0, 1, 0]), axis=1)
# datasetParams['triangle_intersect_ind'] = datasetParams.apply(
#    lambda r: argrelextrema(np.array(f.get(r['key']))[r['N_spins'] // 2, 0, :], np.less)[0][0] // 3, axis=1)
datasetParams.drop(columns=['beta', 'Initial state'], inplace=True)
# datasetParams['scatter_ind'] = datasetParams.apply(
#    lambda r: int(np.where(np.array(f.get(r['key'])).real[::-1, 1, 0] > 0)[0][0]) - f.get(r['key']).attrs[
#        'N_spins'] // 2,
#    axis=1)
# datasetParams['scatter_fock'] = datasetParams.apply(lambda r: int(np.array(f.get(r['key']))[
#                                                                      int(np.where(
#                                                                          np.array(f.get(r['key']))[:, 1, 0] > 0)[0][
#                                                                              0]), 1, 0].real), axis=1)
# datasetParams['scatter_time'] = datasetParams.apply(
#    lambda r: int(np.where(np.array(f.get(r['key']))[r['scatter_ind'] + r['N_spins'] // 2, 0, :] > 0.05)[0]), axis=1)

# datasetParams['V_vdw'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['V_vdw'], axis=1)
datasetParams.sort_values(['endTime', 'omegaTrap', 'kappa', 'N_phonons', 'Fock', 'r0', 'alpha'],
                          ascending=[False, True, True, True, True, True, True],
                          inplace=True,
                          ignore_index=True,
                          key=abs)
# now discard all values which are not until time 10

datasetParams = datasetParams[
    datasetParams['endTime'] >= 10][
    datasetParams['DeltaEE'] == -500][
    datasetParams['omegaTrap'] == 8][
    datasetParams['Fock'] == 0][
    datasetParams['N_phonons'] == 2][
    datasetParams['key'] != '71146ce5134d455d88c9596796e975c6'][
    datasetParams['key'] != '75f71e12635c4889988029ef20fef15d'][
    datasetParams['key'] != '4bc46d6b56664480b7bdc22541c5def0'][
    # datasetParams['cutoff'] == 1e-5][
    datasetParams['kappa'] == 0
    ]
datasetParams.reset_index(inplace=True, drop=True)
print(datasetParams)
# %% load the datasets chosen above # TODO mask with max-bond
ts = f.get(datasetParams['key'][0]).attrs['times']  # [:50]  # TODO remove time limit again
N_spins = f.get(datasetParams['key'][0]).attrs['N_spins']
Omega = f.get(datasetParams['key'][0]).attrs['Omega']
eOps = []
for index, row in datasetParams.iterrows():
    eOps.append(f.get(row['key']).attrs['observables'])
datasetData = np.zeros([datasetParams.shape[0],
                        N_spins,
                        len(max(eOps, key=len)),
                        len(ts),
                        ], dtype=complex)
for index, row in datasetParams.iterrows():
    data = np.array(f.get(row['key']))[::-1, :, :len(ts)]
    n = f.get(row['key']).attrs['N_spins']
    if n < datasetData.shape[1]:
        # TODO mask the missing values
        start_index = (datasetData.shape[1] - n) // 2
        datasetData[index, start_index:start_index + n, :data.shape[1]] = data
    elif n > datasetData.shape[1]:
        raise IndexError('The array datasetData should have a larger spin dimension. Choose the maximum one!')
    else:
        datasetData[index, :, :data.shape[1]] = data
# %% shift site coordinates to center
if N_spins % 2 == 0:
    sitesCenter = np.array(range(-(N_spins // 2), N_spins // 2))
else:
    sitesCenter = np.array(range(-(N_spins // 2), N_spins // 2 + 1))
# %% specify subplots' parameters
max_N = N_spins // 2  # plot all sites
# max_N = 8
# max_N = 21  # plot only up to site number
subplot_params = {
    'nrows': 1,
    'ncols': 7,
    'squeeze': False,
    'figsize': (500 / 72, 120 / 72),
    'sharex': 'col',
    'sharey': 'row',
    'dpi': 250,
}
# subplot_params['figsize'] = np.array([4.6, 3.6])*0.9

X, Y = np.meshgrid(sitesCenter, ts)


def annotate_plot(ca_, textcolor='white'):
    """
    Adds labels to the given axis. Which annotations are added can be configured with global variables above.
    :param ca_: current axis
    :param textcolor: The textcolor of the annotations
    :return:
    """
    annotation_params_heatmap = {
        'xycoords': 'axes fraction',
        'size': 12,
        'color': textcolor,
        'fontweight': 'medium',
        'path_effects': [path_effects.withStroke(linewidth=0.4, foreground=textcolor)],
    }

    annotation_positions = [
        {'xy': (0.05, 0.82)},
        {'xy': (0.05, 0.62)},
        {'xy': (0.05, 0.38)},
        {'xy': (0.05, 0.2)},
        {'xy': (0.05, 0.08)},
    ]

    ca.set_xlim([-max_N - 0.5, max_N + 0.5])
    # ca.set_xticks([-20, 0, 20])
    #    ca_.annotate(rf'$\kappa={row["kappa"]}\,\Omega$', **annotation_positions[0], **annotation_params_heatmap)
    # ca_.annotate(rf'$r_0={row["r0"]}$', **annotation_positions[3], **annotation_params_heatmap)
    ca_.annotate(fr'$\text{{Fock}} \ket{row["Fock"]}$', **annotation_positions[0], **annotation_params_heatmap)
    #   ca_.annotate(fr'$\ket{{\psi}}_{{{row["scatter_ind"]}}}=\ket{{{row["scatter_fock"]}}}$',
    #                **annotation_positions[2], **annotation_params_heatmap)
    # ca_.annotate(rf'P_dim$={row["N_phonons"]}$', **annotation_positions[3], **annotation_params_heatmap)
    # ca_.annotate(rf'co$={row["cutoff"]}$', **annotation_positions[2], **annotation_params_heatmap)
    if isAlphaState:
        if np.round(row["alpha"]).imag == 0:
            a = np.round(row["alpha"], 2).real
        elif np.round(row["alpha"], 2).real == 0:
            a = np.round(row["alpha"], 2).imag * 1j
        else:
            a = np.round(row["alpha"], 2)
        if np.abs(a) == 0:
            a = 0
        ca_.annotate(rf'$\kappa={row["kappa"]}$', **annotation_positions[1], **annotation_params_heatmap)
    if showPhononTrunc:
        if isAlphaState:
            ca_.annotate(rf'$trunc={row["N_phonons"]}$', **annotation_positions[3], **annotation_params_heatmap)
        else:
            ca_.annotate(rf'$trunc={row["N_phonons"]}$', **annotation_positions[2], **annotation_params_heatmap)
    if ax_col == 0:
        ca_.set_ylabel(r'$\Omega t$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca_.set_xlabel('$j$', usetex=True)


def sup_label_heatmap(ca_, im_, fig_, ax_):
    """
    Creates a colorbar next to subplots and rescales the figure
    :param ca_: last current axis (bottom right)
    :param im_: an image with a colorplot
    :param fig_: the current figure
    :param ax_: the current axes
    :return:
    """
    # ca is always lower right axes
    ca_top_right = ax[0][-1]
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([ca.get_position().x1 + 0.02,
                            ca.get_position().y0,
                            0.01,
                            ca_top_right.get_position().y1 - ca_.get_position().y0])
    fig.colorbar(im, cax=cbar_ax)


# %% plot the Rydberg density in <|exe|>
# datasetParams['scatter_time'] = datasetParams['scatter_ind'] * 0

fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    if 'e' in eOps[i]:
        e_index = np.ndarray.item(np.where(eOps[i] == 'e')[0])
    else:
        continue
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca = ax[ax_row][ax_col]
    im = ca.pcolormesh(X, Y,
                       (np.transpose(
                           datasetData[i, :, e_index, :].real)),
                       cmap='viridis',
                       shading='auto',
                       vmin=0, vmax=1, rasterized=True)
    # scatter_t = int(np.where(datasetData[i, row['scatter_ind'] + row['N_spins'] // 2, 0, :] > 0.05)[0][0])
    # datasetParams.at[i, 'scatter_time'] = ts[scatter_t]
    # ca.axhline(ts[row['triangle_intersect_ind']], color='red')
    # ca.axhline(datasetParams.at[i, 'scatter_time'], linestyle=':', color='C1', linewidth=1)
    # ca.axvline(row['scatter_ind'], linestyle='--', color='red', linewidth=1)
    # ca.axvline(row['scatter_ind'], linestyle=':', color='red', linewidth=2)
    annotate_plot(ca)
    ca.annotate(
        rf'$n_j$', (0.7, 0.1), xycoords='axes fraction', size=12, color="white",
        path_effects=[path_effects.withStroke(linewidth=0.4, foreground='white')])
    # ca.yaxis.set_minor_locator(MultipleLocator(0.5))
    ca.yaxis.set_major_locator(MultipleLocator(2))
    # ca.set_ylim([0, 10])
    # ca.set_yticks([0, 5, 10])
    # ca.set_xlim([-25, 25])
    # ca.set_xticks([-20,-10, 0,10, 20])
# fig.suptitle(r'Rydberg $\ket{0}$')
sup_label_heatmap(ca, im, fig, ax)
# ax[0][0].set_yticks([0,5,10,15])
# plt.savefig(f'./{subfolder}/Rydberg-density.png')
# plt.savefig(f'./{subfolder}/Rydberg-density.pdf', dpi=600)
if showImage:
    plt.show()
plt.close()

# %% fit the initial triangle
fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=400)
r0s = []
minima = []
for i, row in datasetParams.iterrows():
    if 'e' in eOps[i]:
        e_index = np.ndarray.item(np.where(eOps[i] == 'e')[0])
    else:
        continue
    vals = (datasetData[i, N_spins // 2, e_index, :].real - datasetData[i, N_spins // 2, e_index, -1].real) / (
            datasetData[i, N_spins // 2, e_index, 0].real - datasetData[i, N_spins // 2, e_index, -1].real)
    plt.plot(ts, vals, '.-', label=f'$r_0={row["r0"]}$')
    mins = np.where(vals <= 0.5)[0]
    r0s += [row['r0']]
    minima += [ts[mins[0]]]
    plt.plot(ts[mins[0]], vals[mins[0]].real, 'x', color=f'C{i}')
    # plt.axvline((row['r0']) / 3, color=f'C{i}')
plt.xlabel('$\Omega t$')
plt.ylabel('$n_{N/2}$')
plt.tight_layout()
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=400)
ax.plot(r0s, minima, 'x', label='$t_0$')
ax.plot(np.linspace(1, 16, 10), (np.linspace(1, 16, 10) - 1) / 2, label='$(r_0-1)/2$')
[slope, inters] = np.polyfit(r0s, minima, deg=1)
r = np.linspace(1, 16, 5)
print(slope * 9 + inters)
ax.plot(r, r * slope + inters, label='fit')
ax.annotate(rf'$\Omega t_0={np.round(slope, 3)} r_0+{np.round(inters, 3)}$', (2, 5))
ax.set_xlabel('$r_0$')
ax.set_ylabel('$\Omega t_0$')
plt.legend()
plt.show()
# %% plot the difference in the density in <|exe|>
if len(datasetParams) == 2 and 'e' in eOps or True:
    e_index = np.ndarray.item(np.where(eOps == 'e')[0])
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=subplot_params['dpi'])
    model_data = np.transpose(datasetData[0, :, e_index, :].real)
    mps_data = np.transpose(datasetData[1, :, e_index, :].real)
    relDifference = (model_data - mps_data)  # / (mps_data + model_data)
    im = ax.pcolormesh(X[:, :], Y[:, :], relDifference[:, :], cmap='viridis', shading='auto')  # ,vmax=0.09,vmin=-0.1)
    ax.set_title(r'Difference in Rydberg density')
    fig.colorbar(im)
    ax.annotate('model-MPS', (-20, 5), color='white')
    fig.tight_layout()
    plt.savefig(f'./{subfolder}/Rydberg-difference.png')
    if showImage:
        plt.show()
    plt.close()

# %% Rydberg density cut at t=8
if len(datasetParams) == 2 and 'e' in eOps or True:
    e_index = np.ndarray.item(np.where(eOps == 'e')[0])
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=subplot_params['dpi'])
    model_data = np.transpose(datasetData[0, :, e_index, 20].real)
    mps_data = np.transpose(datasetData[1, :, e_index, 20].real)
    # mps_data_2 = np.transpose(datasetData[3, :, e_index, 80].real)
    ax.plot(range(N_spins), model_data, 'D', label='model', color='black')
    ax.plot(range(N_spins), mps_data, 'x', label='mps')
    # ax.plot(range(N_spins), mps_data_2, '+', label='mps2')
    # im = ax.pcolormesh(X[:, :], Y[:, :], relDifference[:, :], cmap='viridis', shading='auto',vmax=0.09,vmin=-0.1)
    ax.set_title(rf'Rydberg density at $\Omega t={ts[20]}$')
    ax.legend()
    # fig.colorbar(im)
    ax.annotate('model-MPS', (-20, 5), color='white')
    fig.tight_layout()
    plt.savefig(f'./{subfolder}/Rydberg-comparison.png')
    if showImage:
        plt.show()
    plt.close()

# %% relative deviation cut at t=8
if len(datasetParams) == 2 and 'e' in eOps or True:
    e_index = np.ndarray.item(np.where(eOps == 'e')[0])
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=subplot_params['dpi'])
    model_data = np.transpose(datasetData[0, :, e_index, 74].real)
    mps_data = np.transpose(datasetData[1, :, e_index, 74].real)
    # mps_data_2 = np.transpose(datasetData[3, :, e_index, 80].real)
    ax.plot(range(N_spins), model_data - mps_data, label='model-mps')
    # ax.plot(range(N_spins), model_data - mps_data_2, label='model-mps2')
    # im = ax.pcolormesh(X[:, :], Y[:, :], relDifference[:, :], cmap='viridis', shading='auto',vmax=0.09,vmin=-0.1)
    ax.set_title(r'Difference in Rydberg density')
    ax.legend()
    # fig.colorbar(im)
    ax.annotate('model-MPS', (-20, 5), color='white')
    fig.tight_layout()
    plt.savefig(f'./{subfolder}/Rydberg-difference.png')
    if showImage:
        plt.show()
    plt.close()
# %% plot the Rydberg density in <|rxr|>
if 'r' in eOps:
    r_index = np.ndarray.item(np.where(eOps == 'r')[0])
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(X, Y,
                           np.transpose(datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, r_index, :].real),
                           cmap='viridis', vmin=0, vmax=1, shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'Local Rydberg density $\ev{\dyad{r}_j}$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/Rydberg-density-rxr.png')
    if showImage:
        plt.show()
    plt.close()

# %% plot the Rydberg densities of the initial triangle
if 'e' in eOps:
    e_index = np.ndarray.item(np.where(eOps == 'e')[0])
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        if even_state:
            triangle_sites = np.arange(-row['r0'] // 2, row['r0'] // 2 + 1)
        else:
            triangle_sites = np.arange(-row['r0'] // 2 + 1, row['r0'] // 2 + 1)
        for j in triangle_sites:
            loc_density = datasetData[i, N_spins // 2 + j, 0, :].real
            ca.plot(ts, loc_density, label=f'j={j}')
            try:
                local_min = argrelextrema(loc_density, np.less)[0][0]
                ca.plot(ts[local_min], loc_density[local_min], 'rx')
            except IndexError:
                print(f'no minimum found in dataset {i}.')
        ca.axvline(ts[row['triangle_intersect_ind']])
        if ax_col == 0:
            ca.set_ylabel('$n$')
        if ax_row == ax.shape[0] - 1:
            ca.set_xlabel(r'time in $\Omega t$')
        if ax_row == ax.shape[0] - 1 and ax_col == 0:
            ca.legend(loc='lower center', ncols=row['r0'], bbox_to_anchor=(1.7, 0.9))
        # annotate_plot(ca)
    fig.suptitle(r'Relaxation of excited Rydberg atoms')
    # sup_label_heatmap(ca, im, fig, ax)
    fig.tight_layout()
    plt.savefig(f'./{subfolder}/Triangle-equilibration.png')
    plt.savefig(f'./{subfolder}/Triangle-equilibration.pdf')
    if showImage:
        plt.show()
    plt.close()

# %% plot the Phonon density
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    if 'n_a' in eOps[i]:
        ada_index = np.ndarray.item(np.where(eOps[i] == 'n_a')[0])
    else:
        continue
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca = ax[ax_row][ax_col]
    im = ca.pcolormesh(X, Y, np.transpose(
        datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, ada_index, :].real),
                       cmap='viridis',
                       vmin=np.amin(datasetData[:, :, ada_index, :].real),
                       vmax=np.amax(datasetData[:, :, ada_index, :].real),
                       shading='auto',
                       )
    # ca.axvline(row['scatter_ind'], linestyle='--', color='red', linewidth=0.5)
    annotate_plot(ca)
# fig.suptitle(r'Local Phonon occupation number $a^\dagger_j a_j$')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/Phonon-density.png')
plt.savefig(f'./{subfolder}/Phonon-density.pdf')
if showImage:
    plt.show()
plt.close()

# %% plot the positions in the traps
if 'a' in eOps and 'a_dag' in eOps:
    a_index = np.ndarray.item(np.where(eOps == 'a')[0])
    a_dag_index = np.ndarray.item(np.where(eOps == 'a_dag')[0])
    mu_1 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_index, :] \
           + datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_dag_index, :]
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(
            X, Y,
            np.transpose(mu_1[i].real),
            cmap='viridis',
            vmin=np.amin(mu_1.real),
            vmax=np.amax(mu_1.real),
            shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'position in trap relative to center: $a_j+a_j^\dagger$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/position.png')
    if showImage:
        plt.show()
    plt.close()

# %% plot the second cumulant of the positions in the traps
if 'a' in eOps and 'a_dag' in eOps and 'xa^2' in eOps:
    a_index = np.ndarray.item(np.where(eOps == 'a')[0])
    a_dag_index = np.ndarray.item(np.where(eOps == 'a_dag')[0])
    xa_sq_index = np.ndarray.item(np.where(eOps == 'xa^2')[0])
    mu_1 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_index, :] \
           + datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_dag_index, :]
    mu_2 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, xa_sq_index, :]
    second_cumulant = mu_2  # - np.power(mu_1, 2)
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(
            X, Y,
            np.transpose(second_cumulant[i].real),
            cmap='viridis',
            vmin=np.amin(second_cumulant.real),
            vmax=np.amax(second_cumulant.real),
            shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'spatial 2nd cumulant in trap $\mu_2(x)$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/spatial-spread.png')
    if showImage:
        plt.show()
    plt.close()

# %% plot the third cumulant of the positions in the traps
if 'a' in eOps and 'a_dag' in eOps and 'xa^2' in eOps and 'xa^3' in eOps:
    a_index = np.ndarray.item(np.where(eOps == 'a')[0])
    a_dag_index = np.ndarray.item(np.where(eOps == 'a_dag')[0])
    xa_sq_index = np.ndarray.item(np.where(eOps == 'xa^2')[0])
    xa_3_index = np.ndarray.item(np.where(eOps == 'xa^3')[0])
    mu_1 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_index, :] \
           + datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, a_dag_index, :]
    mu_2 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, xa_sq_index, :]
    mu_3 = datasetData[:, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, xa_3_index, :]
    third_cumulant = mu_3 - 3 * mu_2 * mu_1 + 2 * np.power(mu_1, 3)
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(
            X, Y,
            np.transpose(third_cumulant[i].real),
            cmap='viridis',
            vmin=np.amin(third_cumulant.real),
            vmax=np.amax(third_cumulant.real),
            shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'spatial 3rd cumulant in trap: $\mu_3(x)$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/spatial-3rd-cumulant.png')
    if showImage:
        plt.show()
    plt.close()

# %% Calculate the sum over the densities (we need them for normalization)
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    if 'e' in eOps[i] and 'n_a' in eOps[i]:
        e_index = np.ndarray.item(np.where(eOps[i] == 'e')[0])
        ada_index = np.ndarray.item(np.where(eOps[i] == 'n_a')[0])
    else:
        continue
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]
    ax2[ax_row][ax_col] = ca1.twinx()
    ca2 = ax2[ax_row][ax_col]
    ca2.sharey(ax2[ax_row][0])
    ca1.plot(ts, normalizationFunction[e_index], '.-', label='Rydbergs')
    # ca1.axvline(row['scatter_time'], linestyle='--', color='red', linewidth=1)
    if 'r' in eOps[i]:
        ryd_sum = normalizationFunction[e_index] + normalizationFunction[r_index]
        ca1.plot(ts, ryd_sum, '.-', label='Rydbergs')
    # ca1.set_ylim([0, 15])
    # ca2.set_ylim([100, 105])
    ca2.plot(ts, normalizationFunction[ada_index], label='Phonons', color='C2')
    annotate_plot(ca2, textcolor='black')
    # FIXME y label wrong for multiple rows
    if ax_col == 0:
        ca1.set_ylabel('$n_e,n_r$')
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$')
        ca2.set_ylabel('')
    if ax_col == ax.shape[1] - 1:
        ca2.set_ylabel('$a^\dagger a$', color='C2')
    if ax_col < ax.shape[1] - 1:
        plt.setp(ca2.get_yticklabels(), visible=False)
# fig.suptitle('Extensive observables')
plt.tight_layout()
plt.savefig(f'./{subfolder}/sum-density.png')
plt.savefig(f'./{subfolder}/sum-density.pdf')
if showImage:
    plt.show()
plt.close()

# %% plots for THz paper

fig = plt.figure(dpi=250, figsize=(3.667, 1.3))
gs = GridSpec(2, 2, figure=fig)
fig.subplots_adjust(left=0.096, bottom=0.25, right=0.925, top=0.96, wspace=0.4)
cax = plt.axes([0.45, 0.25, 0.015, 0.96 - 0.25])

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[:, 1])

im = ax0.pcolormesh(Y, X,
                    np.transpose(datasetData[1, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, e_index, :].real),
                    cmap='viridis', vmin=0, vmax=1, rasterized=True)
ax0.annotate(f'$\kappa={datasetParams["kappa"][1]}\Omega$', xy=(0.55, 0.7), xycoords='axes fraction', color='white')
ax0.set_ylabel('$j$')
ax0.set_xlim([0, ts[-1]])
ax0.yaxis.labelpad = -4
ax0.yaxis.set_minor_locator(MultipleLocator(1))
ax0.annotate('$\mathcal{S}_j$', xy=(0.1, 0.15), xycoords='axes fraction', color='white')
ax0.xaxis.set_minor_locator(MultipleLocator(1))
ax0.xaxis.set_major_locator(MultipleLocator(5))
ax0.yaxis.set_major_locator(MultipleLocator(5))
ax0.set_xticklabels([])
ax0.set_xlabel('')

ax1.pcolormesh(Y, X,
               np.transpose(datasetData[2, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, e_index, :].real),
               cmap='viridis', vmin=0, vmax=1, rasterized=True)
ax1.annotate(f'$\kappa={datasetParams["kappa"][2]}\Omega$', xy=(0.55, 0.7), xycoords='axes fraction', color='white')
ax1.set_xlabel(r'$\Omega t$')
ax1.set_xlim([0, ts[-1]])
ax1.xaxis.labelpad = 0
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.set_ylabel('$j$')
ax1.yaxis.labelpad = -4
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(5))
ax1.annotate('$\mathcal{S}_j$', xy=(0.1, 0.15), xycoords='axes fraction', color='white')

ax2.plot(ts, np.sum(datasetData[0].real, axis=0)[e_index], '-', label=f'$\kappa={datasetParams["kappa"][0]}$')
ax2.plot(ts, np.sum(datasetData[1].real, axis=0)[e_index], '-', label=f'$\kappa={datasetParams["kappa"][1]}$')
ax2.plot(ts, np.sum(datasetData[2].real, axis=0)[e_index], '-', label=f'$\kappa={datasetParams["kappa"][2]}$')
ax2.set_xlabel(r'$\Omega t$')
ax2.set_xlim([0, ts[-1]])
ax2.set_xticks([0, 5, 10])
ax2.xaxis.labelpad = 0
ax2.set_ylabel(r'$\mathcal{S}$')
ax2.set_ylim([0, 8.5])
ax2.set_yticks([0, 5])
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.yaxis.labelpad = 0
ax2.yaxis.set_label_coords(1.13, .45)
ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(1))
ax2.annotate(r'$\kappa_{0.0}$', xy=(0.8, 0.9), xycoords='axes fraction', color='C0')
ax2.annotate(rf'$\Upsilon={np.round(np.amax(np.sum(datasetData[0].real, axis=0)[e_index]), 1)}$', xy=(0.02, 0.87),
             xycoords='axes fraction', color='C0')
ax2.annotate(r'$\kappa_{1.5}$', xy=(0.8, 0.77), xycoords='axes fraction', color='C1')
ax2.annotate(rf'$\Upsilon={np.round(np.amax(np.sum(datasetData[1].real, axis=0)[e_index]), 1)}$', xy=(0.24, 0.45),
             xycoords='axes fraction', color='C1')
ax2.annotate(r'$\kappa_{3.0}$', xy=(0.8, 0.62), xycoords='axes fraction', color='C2')
ax2.annotate(rf'$\Upsilon={np.round(np.amax(np.sum(datasetData[2].real, axis=0)[e_index]), 1)}$', xy=(0.09, 0.07),
             xycoords='axes fraction', color='C2')

ax0.annotate('(a)', xy=(0.85, 0.14), xycoords='axes fraction', color='white')
ax1.annotate('(b)', xy=(0.85, 0.14), xycoords='axes fraction', color='white')
ax2.annotate('(c)', xy=(0.85, 0.07), xycoords='axes fraction', color='black')

fig.colorbar(im, ax=ax[0], cax=cax, location='right', pad=0.0)

fig.savefig(f'./{subfolder}/VibrationLessFacilitation.pdf', dpi=600)

plt.show()
# %% Plot the first moment time evolution --> center of mass movement
FIT_PERCENTILE = 70
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    normalizedDensities = datasetData[i].real / normalizationFunction
    moment1st = np.tensordot(normalizedDensities, sitesCenter, axes=([0], [0]))
    r0 = row['r0']

    fitIndex = int(len(ts) - len(ts) * (1 - (100 - FIT_PERCENTILE) / 100))
    y1st = abs(moment1st[e_index]) - ((r0 - 1) % 2) / 2
    [slope_before, intersect_before], cov = np.polyfit(ts[fitIndex:], y1st[fitIndex:], deg=1, cov=True)
    print(cov)
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]
    # ca1.axvline(row['scatter_time'], linestyle='--', color='red', linewidth=1)
    ca1.plot(ts, y1st, '.-', label='1st Moment', color='C0')
    ca1.plot(ts, ts * slope_before + intersect_before, '--', label='fit 1st Moment', color='C1')
    ca1.annotate(rf'$\mu_1={np.round(slope_before, 2)}t+{{{np.round(intersect_before, 3)}}}$', (0.05, 0.1),
                 xycoords='axes fraction', size=10, color="C1",
                 path_effects=[path_effects.withStroke(linewidth=0.4, foreground='C1')])
    # ca1.set_ylim([0.0001, 0.005])
    annotate_plot(ca1, textcolor='black')
    ca1.set_xlim([0.0, ts[-1]])
    if ax_col == 0:
        ca1.set_ylabel(r'$\mu_1$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$')
    # ca1.set_xticks(np.linspace(0, 10, 6))
    ca1.grid(visible=True)
# fig.suptitle('Center of mass Rydberg density')
plt.tight_layout()
plt.savefig(f'./{subfolder}/1st-moment1-scatter.png')
plt.savefig(f'./{subfolder}/1st-moment1-scatter.pdf')
if showImage:
    plt.show()
plt.close()

# %% Plot the second moment time evolution
FIT_PERCENTILE = 20
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
subplot_params_loc = subplot_params.copy()
subplot_params['sharey'] = 'all'
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    normalizedDensities = datasetData[i].real / normalizationFunction
    r0 = row['r0']
    kap = row['kappa']
    moment2nd = np.tensordot(normalizedDensities, sitesCenter ** 2, axes=([0], [0])) \
                - np.tensordot(normalizedDensities, sitesCenter ** 1, axes=([0], [0])) ** 2 \
                - (r0 ** 2 - 1) / 12

    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]

    ca1.plot(ts, moment2nd[e_index], '.-', label='2nd Moment', color='C0')
    # ca1.plot(ts[6:fitIndex], moment2nd[e_index][6:fitIndex], '.-', label='2nd Moment', color='C2')
    # TODO add errorbars from asymmetry
    # now we fit the 2nd moment to a power law. We use here the last FIT_PERCENTILE of the datapoints
    [slope_before, intersect_before] = np.polyfit(np.log(ts[2:]), np.log(moment2nd[e_index][2:]),
                                                  deg=1)
    # [slope_after, intersect_after] = np.polyfit(np.log(ts[fitIndex + 10:]),
    #                                            np.log(moment2nd[e_index][fitIndex + 10:]),
    #                                            deg=1)
    logTs = np.logspace(-2, np.log(15), 100)
    ca1.plot(logTs, logTs ** slope_before * np.exp(intersect_before), '--', label='fit', color='C1')
    ca1.annotate(
        rf'$\propto{np.round(np.exp(intersect_before), 2)}t^{{{np.round(slope_before, 2)}}}$',
        (0.2, 0.05),
        xycoords='axes fraction', size=10, color="C1",
        path_effects=[path_effects.withStroke(linewidth=0.4, foreground='C1')])

    # ca1.plot(logTs, logTs ** slope_after * np.exp(intersect_after), '--', label='fit', color='C3')
    # ca1.annotate(
    #    rf'$\propto{np.round(np.exp(intersect_after), 2)}t^{{{np.round(slope_after, 2)}}}$',
    #   (0.55, 0.4),
    #   xycoords='axes fraction', size=10, color="C3",
    #   path_effects=[path_effects.withStroke(linewidth=0.4, foreground='C3')])
    ca1.set_xscale('log')
    ca1.set_yscale('log')
    ca1.set_ylim([0.1, 250])
    ca1.set_xlim([0.2, 10])
    annotate_plot(ca1, textcolor='black')
    if ax_col == 0:
        ca1.set_ylabel(r'$\delta\sigma(t)$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$', usetex=True)
# fig.suptitle('Second moment of Rydberg densities')
plt.tight_layout()
plt.savefig(f'./{subfolder}/2nd-moment.png')
plt.savefig(f'./{subfolder}/2nd-moment.pdf')
if showImage:
    plt.show()
plt.close()

# %% Compare the second moment time evolution
try:
    moment2nd_0 + moment2nd_1
except:
    print('moments not defined')
    pass
else:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2), dpi=300)
    ca1 = ax
    # ca1.plot(ts, moment2nd_0[0], '.')
    # ca1.plot(ts, moment2nd_1[0], 'x')
    ca1.plot(ts[:], (moment2nd_0[0, :] - moment2nd_1[0, :]))  # / (moment2nd_0[0, :] + moment2nd_1[0, :]), 'x')

    # ca1.set_xscale('log')
    # ca1.set_yscale('log')
    # ca1.set_ylim([-1, 2])
    ca1.set_xlim([0, 5])
    # annotate_plot(ca1, textcolor='black')
    ca1.set_ylabel(r'$\mu_2(t)$', usetex=True)

    ca1.set_xlabel(r'time in $\Omega t$', usetex=True)
    # fig.suptitle('Second moment of Rydberg densities')
    plt.tight_layout()
    plt.savefig(f'./{subfolder}/2nd-moment-difference.png')
    if showImage:
        plt.show()
    plt.close()

# %% plot the entropy calculated as bipartite entanglement entropy of left side with N_spins//2
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    if 'entropy' in eOps[i]:
        entropy_index = np.ndarray.item(np.where(eOps[i] == 'entropy')[0])
        print(entropy_index)
    else:
        continue
    # entropy_index = np.ndarray.item(np.where(f.get(row['key']).attrs['observables'] == 'entropy')[0])
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]

    nextToCenter = 1
    for site in range(-nextToCenter, nextToCenter + 1):
        ca1.plot(ts, datasetData[i, N_spins // 2 + site, entropy_index, :].real, '.', label=rf'$j={site}$')
    # ca1.axvline(ts[row['triangle_intersect_ind']])
    annotate_plot(ca1, textcolor='black')
    if ax_col == 0:
        ca1.set_ylabel('entropy')
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$')
    ca1.xaxis.set_major_locator(MultipleLocator(1))
    ca1.yaxis.set_major_locator(MultipleLocator(1))
    ca1.xaxis.set_minor_locator(MultipleLocator(0.25))
    ca1.yaxis.set_minor_locator(MultipleLocator(0.25))
    # ca1.set_ylim([0.5, 5])
    # ca1.set_xlim([1, 10])
    # ca1.set_xscale('log')
    # ca1.set_yscale('log')
# fig.suptitle('Bipartite Entanglement Entropy')
plt.legend()
plt.tight_layout()
plt.savefig(f'./{subfolder}/entanglement.png')
if showImage:
    plt.show()
plt.close()

# %%
# TODO plot the Rydberg shapes at different times (cut at fixed time through Rydberg density plot)
# TODO plot the Phonon shapes at different times (cut at fixed time through Phonon density plot)

# %%##############################################################################################################
# NUMERICAL CHECKS
##################################################################################################################

# check the bond-dimension of the sites. Remember, this is the bond dimension of the left side of the site
fig, ax = plt.subplots(**subplot_params)
X, Y = np.meshgrid(sitesCenter[N_spins // 2 - max_N:N_spins // 2 + max_N + 1], ts)
for i, row in datasetParams.iterrows():
    if 'bond_dim' in eOps[i]:
        bond_dim_index = np.ndarray.item(np.where(eOps[i] == 'bond_dim')[0])
    else:
        continue
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca = ax[ax_row][ax_col]
    im = ca.pcolormesh(X, Y,
                       np.transpose(
                           datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, bond_dim_index, :].real),
                       cmap='viridis',
                       vmin=-1,
                       vmax=45,
                       shading='auto')
    annotate_plot(ca)
    ca.yaxis.set_major_locator(MultipleLocator(1))
    ca.yaxis.set_minor_locator(MultipleLocator(0.25))
# fig.suptitle('Bond dimension')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/bond-dimensions.png')
if showImage:
    plt.show()
plt.close()

# %% Calculate the asymmetry of the measured observables

leftSite = N_spins // 2 - max_N
if even_state:
    leftSite += 1
rightSite = N_spins // 2 + max_N + 1

X, Y = np.meshgrid(sitesCenter[leftSite:rightSite], ts)
asymmetry = np.empty([datasetParams.shape[0]] + list(np.transpose(datasetData[0, leftSite:rightSite, :, :]).shape))
for i, row in datasetParams.iterrows():
    lightCone = np.transpose(datasetData[i, leftSite:rightSite, :, :].real)
    asymmetry[i] = lightCone - np.flip(np.copy(lightCone), axis=-1)

# %% check the asymmetry in the Rydberg density. This is an indicator of wrong simulation results,
# because the time-evolution (light-cone) of the system should be symmetric

fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca = ax[ax_row][ax_col]

    im = ca.pcolormesh(X, Y, asymmetry[i, :, 0],
                       cmap='BrBG',
                       # vmin=np.amin(asymmetry[:, :, 0]),
                       # vmax=np.amax(asymmetry[:, :, 0]),
                       shading='auto',
                       # norm=colors.SymLogNorm(vmin=-0.2,linthresh=0.1)
                       rasterized=True,
                       )
    annotate_plot(ca, textcolor='black')
# fig.suptitle('Asymmetry in Rydberg density')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/asymmetry-rydbergs.png')
plt.savefig(f'./{subfolder}/asymmetry-rydbergs.pdf', dpi=600)
if showImage:
    plt.show()
plt.close()

# %% check the asymmetry in the Phonon density. This is an indicator of wrong simulation results,
# because the time-evolution (light-cone) of the system should be symmetric

fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca = ax[ax_row][ax_col]

    im = ca.pcolormesh(X, Y, asymmetry[i, :, 1], cmap='BrBG',
                       vmin=np.amin(asymmetry[:, :, 1]),
                       vmax=np.amax(asymmetry[:, :, 1]),
                       shading='auto')
    annotate_plot(ca, textcolor='black')
# fig.suptitle('Asymmetry in Phonon density')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/asymmetry-phonons.png')
if showImage:
    plt.show()
plt.close()
