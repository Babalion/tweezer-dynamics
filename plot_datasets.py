import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{physics}')

# %% load file and set path

# store the plots in the same folder as the data are
subfolder = 'vee'

# specify if the initial state has an even or odd number of excited Rydbergs
even_state = True
isAlphaState = False
showPhononTrunc = False
showImage = True

# We can plot only datasets with equal times
f = h5py.File(f'./{subfolder}/{subfolder}.h5', 'r')
# %% print list of kappa and omega which are in the dataset
datasetParams = pd.DataFrame(f.keys(), columns=['key'], dtype=str)
datasetParams['endTime'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['times'][-1], axis=1)
datasetParams['omegaTrap'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['omegaTrap'], axis=1)
datasetParams['kappa'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['kappa'], axis=1)
datasetParams['N_phonons'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['N_phonons'], axis=1)
datasetParams['alpha'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['alpha'], axis=1)
datasetParams['beta'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['beta'], axis=1)
# datasetParams['V_vdw'] = datasetParams.apply(lambda r: f.get(r['key']).attrs['V_vdw'], axis=1)
datasetParams.sort_values(['endTime', 'omegaTrap', 'kappa', 'N_phonons', 'alpha', 'beta'],
                          ascending=[False, True, True, True, True, True],
                          inplace=True,
                          ignore_index=True,
                          key=abs)
# now discard all values which are not until time 10
datasetParams = datasetParams[datasetParams['endTime'] >= 10]  # [datasetParams['N_phonons'] == 4]
datasetParams.reset_index(inplace=True)
print(datasetParams)
# %% load the datasets chosen above
# TODO mask with max-bond
ts = f.get(datasetParams['key'][0]).attrs['times']  # [:130]  # TODO remove time limit again
eOps = f.get(datasetParams['key'][0]).attrs['observables']
N_spins = f.get(datasetParams['key'][0]).attrs['N_spins']
Omega = f.get(datasetParams['key'][0]).attrs['Omega']
datasetData = np.empty([datasetParams.shape[0],
                        N_spins,
                        len(eOps),
                        len(ts),
                        ], dtype=complex)
for index, row in datasetParams.iterrows():
    datasetData[index, :, :] = np.array(f.get(row['key']))[:, :, :len(ts)]
# %% shift site coordinates to center
if N_spins % 2 == 0:
    sitesCenter = np.array(range(-(N_spins // 2) + 1, N_spins // 2))
else:
    sitesCenter = np.array(range(-(N_spins // 2), N_spins // 2 + 1))
# %% specify subplots' parameters
max_N = N_spins // 2  # plot all sites
# max_N = 8
# max_N = 21  # plot only up to site number
subplot_params = {
    'nrows': 1,
    'ncols': 1,
    'squeeze': False,
    # size is automatically calculated below
    'sharex': 'col',
    'sharey': 'row',
    'dpi': 400,
}
subplot_params['figsize'] = (subplot_params['ncols'] * 1.8, subplot_params['nrows'] * 1.6)

X, Y = np.meshgrid(sitesCenter[N_spins // 2 - max_N:N_spins // 2 + max_N + 1], ts)


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
        {'xy': (0.1, 0.85)},
        {'xy': (0.09, 0.65)},
        {'xy': (0.1, 0.45)},
        {'xy': (0.1, 0.25)},
    ]

    ca.set_xlim([-max_N - 0.5, max_N + 0.5])
    ca_.annotate(rf'$\kappa={np.round(row["kappa"], 2)}\Omega$', **annotation_positions[0], **annotation_params_heatmap)
    ca_.annotate(rf'$\omega={np.round(row["omegaTrap"], 2)}\Omega$', **annotation_positions[1],
                 **annotation_params_heatmap)
    # ca_.annotate(rf'$V_\mathrm{{VdW}}={np.round(row["V_vdw"], 2)}\Omega$', **annotation_positions[1],
    #             **annotation_params_heatmap)
    if isAlphaState:
        if np.round(row["alpha"]).imag == 0:
            a = np.round(row["alpha"], 2).real
        elif np.round(row["alpha"], 2).real == 0:
            a = np.round(row["alpha"], 2).imag * 1j
        else:
            a = np.round(row["alpha"], 2)
        if np.abs(a) == 0:
            a = 0
        ca_.annotate(rf'$\alpha={a}$'.replace('j', 'i'), **annotation_positions[2], **annotation_params_heatmap)
    if showPhononTrunc:
        if isAlphaState:
            ca_.annotate(rf'$trunc={row["N_phonons"]}$', **annotation_positions[3], **annotation_params_heatmap)
        else:
            ca_.annotate(rf'$trunc={row["N_phonons"]}$', **annotation_positions[2], **annotation_params_heatmap)
    if ax_col == 0:
        ca_.set_ylabel(r'time in $\Omega t$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca_.set_xlabel('site $j$', usetex=True)


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
    fig.subplots_adjust(right=0.98 - 0.2 / ax.shape[1])
    cbar_ax = fig.add_axes([ca.get_position().x1 + 0.02,
                            ca.get_position().y0,
                            0.01,
                            ca_top_right.get_position().y1 - ca_.get_position().y0])
    fig.colorbar(im, cax=cbar_ax)


# %% plot the Rydberg density in <|exe|>
if 'e' in eOps:
    e_index = np.ndarray.item(np.where(eOps == 'e')[0])
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(X, Y,
                           np.transpose(datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, e_index, :].real),
                           cmap='viridis', vmin=0, vmax=1, shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'Local Rydberg density $n_j$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/Rydberg-density.png')
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

# %% plot the Phonon density
if 'n_a' in eOps:
    ada_index = np.ndarray.item(np.where(eOps == 'n_a')[0])
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(X, Y, np.transpose(
            datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, ada_index, :].real),
                           cmap='viridis', vmin=np.amin(datasetData[:, :, 1, :].real),
                           vmax=np.amax(datasetData[:, :, 1, :].real), shading='auto')
        annotate_plot(ca)
    # fig.suptitle(r'Local Phonon occupation number $a^\dagger_j a_j$')
    sup_label_heatmap(ca, im, fig, ax)
    plt.savefig(f'./{subfolder}/Phonon-density.png')
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
    second_cumulant = mu_2 - np.power(mu_1, 2)
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
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]
    ax2[ax_row][ax_col] = ca1.twinx()
    ca2 = ax2[ax_row][ax_col]
    ca2.sharey(ax2[ax_row][0])
    ca1.plot(ts, normalizationFunction[e_index], label='Rydbergs')
    if 'r' in eOps:
        ryd_sum = normalizationFunction[e_index] + normalizationFunction[r_index]
        ca1.plot(ts, ryd_sum, label='Rydbergs')
    # ca1.set_ylim([0, 15])
    # ca2.set_ylim([100, 105])
    ca2.plot(ts, normalizationFunction[ada_index], label='Phonons', color='C2')
    annotate_plot(ca2, textcolor='black')
    if ax_col == 0:
        ca1.set_ylabel('sum of Rydbergs', color='C0')
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$')
    if ax_col == ax.shape[1] - 1:
        ca2.set_ylabel('sum of Phonons', color='C2')
    if ax_col < ax.shape[1] - 1:
        plt.setp(ca2.get_yticklabels(), visible=False)
# fig.suptitle('Extensive observables')
plt.tight_layout()
plt.savefig(f'./{subfolder}/sum-density.png')
if showImage:
    plt.show()
plt.close()

# %% Plot the first moment time evolution --> center of mass movement
# TODO add 1st moment of phonons
# TODO add shared second axis for Phonons
# FIXME y label wrong for multiple rows
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    normalizedDensities = datasetData[i].real / normalizationFunction
    moment1st = np.tensordot(normalizedDensities, sitesCenter, axes=([0], [0]))
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]
    ca1.plot(ts, moment1st[0], '.-', label='1st Moment', color='C0')
    ca1.set_xlim([0.01, ts[-1]])
    annotate_plot(ca1, textcolor='black')
    if ax_col == 0:
        ca1.set_ylabel(r'$\mu_1(t)$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$')
# fig.suptitle('Center of mass Rydberg density')
plt.tight_layout()
plt.savefig(f'./{subfolder}/1st-moment.png')
if showImage:
    plt.show()
plt.close()

# %% Plot the second moment time evolution
# TODO add 2nd moment of phonons
# TODO add shared second axis for Phonons
# FIXME y label wrong for multiple rows
FIT_PERCENTILE = 50
ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
fig, ax = plt.subplots(**subplot_params)
for i, row in datasetParams.iterrows():
    normalizationFunction = np.sum(datasetData[i].real, axis=0)
    normalizedDensities = datasetData[i].real / normalizationFunction
    moment2nd = np.tensordot(normalizedDensities, sitesCenter ** 2, axes=([0], [0]))
    ax_row = i // ax.shape[1]
    ax_col = i % ax.shape[1]
    ca1 = ax[ax_row][ax_col]
    ca1.plot(ts, moment2nd[r_index], '.-', label='2nd Moment', color='C0')
    # TODO add errorbars from asymmetry
    # now we fit the 2nd moment to a power law. We use here the last FIT_PERCENTILE of the datapoints
    fitIndex = int(len(ts) - len(ts) * (1 - (100 - FIT_PERCENTILE) / 100))
    [slope, intersect], cov = np.polyfit(np.log(ts[fitIndex:]), np.log(moment2nd[r_index][fitIndex:]), deg=1, cov=True)
    print(cov)
    logTs = np.logspace(-2, np.log(15), 100)
    ca1.plot(logTs, logTs ** slope * np.exp(intersect), '--', label='fit', color='C1')
    ca1.annotate(rf'$\mu_2={np.round(np.exp(intersect), 1)}t^{{{np.round(slope, 1)}}}$', (0.08, 0.5),
                 xycoords='axes fraction', size=10, color="C1",
                 path_effects=[path_effects.withStroke(linewidth=0.4, foreground='C1')])

    ca1.set_xscale('log')
    ca1.set_yscale('log')
    ca1.set_ylim([1, 100])
    ca1.set_xlim([0.5, 15])
    annotate_plot(ca1, textcolor='black')
    if ax_col == 0:
        ca1.set_ylabel(r'$\mu_2(t)$', usetex=True)
    if ax_row == ax.shape[0] - 1:
        ca1.set_xlabel(r'time in $\Omega t$', usetex=True)
# fig.suptitle('Second moment of Rydberg densities')
plt.tight_layout()
plt.savefig(f'./{subfolder}/2nd-moment.png')
if showImage:
    plt.show()
plt.close()

# %% plot the entropy calculated as bipartite entanglement entropy of left side with N_spins//2
if 'entropy' in eOps:
    entropy_index = np.ndarray.item(np.where(eOps == 'entropy')[0])
    ax2 = [[plt.axis] * subplot_params['ncols']] * subplot_params['nrows']
    fig, ax = plt.subplots(**subplot_params)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca1 = ax[ax_row][ax_col]

        nextToCenter = 1
        for site in range(-nextToCenter, nextToCenter + 1):
            ca1.plot(ts, datasetData[i, N_spins // 2 + site, entropy_index, :].real, '.', label=rf'$j={site}$')
        annotate_plot(ca1, textcolor='black')
        if ax_row == 0 and ax_col == 0:
            ca1.set_ylabel('entropy')
        if ax_row == ax.shape[0] - 1:
            ca1.set_xlabel(r'time in $\Omega t$')
        ca1.set_ylim([0.5, 5])
        ca1.set_xlim([1, 10])
        ca1.set_xscale('log')
        ca1.set_yscale('log')
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
if 'bond_dim' in eOps:
    bond_dim_index = np.ndarray.item(np.where(eOps == 'bond_dim')[0])
    fig, ax = plt.subplots(**subplot_params)
    X, Y = np.meshgrid(sitesCenter[N_spins // 2 - max_N:N_spins // 2 + max_N + 1], ts)
    for i, row in datasetParams.iterrows():
        ax_row = i // ax.shape[1]
        ax_col = i % ax.shape[1]
        ca = ax[ax_row][ax_col]
        im = ca.pcolormesh(X, Y,
                           np.transpose(
                               datasetData[i, N_spins // 2 - max_N:N_spins // 2 + max_N + 1, bond_dim_index, :].real),
                           cmap='viridis',
                           vmin=np.amin(datasetData[:, :, bond_dim_index, :].real - 1),
                           vmax=np.amax(datasetData[:, :, bond_dim_index, :].real + 1),
                           shading='auto')
        annotate_plot(ca)
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

    im = ca.pcolormesh(X, Y, asymmetry[i, :, 0], cmap='viridis',
                       vmin=np.amin(asymmetry[:, :, 0]),
                       vmax=np.amax(asymmetry[:, :, 0]),
                       shading='auto')
    annotate_plot(ca)
# fig.suptitle('Asymmetry in Rydberg density')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/asymmetry-rydbergs.png')
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

    im = ca.pcolormesh(X, Y, asymmetry[i, :, 1], cmap='viridis',
                       vmin=np.amin(asymmetry[:, :, 1]),
                       vmax=np.amax(asymmetry[:, :, 1]),
                       shading='auto')
    annotate_plot(ca)
# fig.suptitle('Asymmetry in Phonon density')
sup_label_heatmap(ca, im, fig, ax)
plt.savefig(f'./{subfolder}/asymmetry-phonons.png')
if showImage:
    plt.show()
plt.close()
