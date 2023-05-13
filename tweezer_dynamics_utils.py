import warnings
from itertools import product

import numpy as np
import quimb.tensor as qtn
import qutip as qt


# %% Define the initial states as classes with inheritance
class InitialState:
    """
    This is a baseclass for an initial MPS state.
    """

    def __init__(self, spin_dim=2, num_spins=1, dim_phonons=1, is_cyclic=False, dtype='complex128'):
        # TODO generalize with **kwargs
        self.spin_dim = spin_dim
        self.num_spins = num_spins
        self.dim_phonons = dim_phonons
        self.__is_cyclic = is_cyclic
        self.__dtype = dtype
        # basis states for spins
        if spin_dim == 2:
            self._g = qt.basis(2, 0)
            self._e = qt.basis(2, 1)
            self._r = None
        elif spin_dim == 3:
            self._g = qt.basis(3, 0)
            self._e = qt.basis(3, 1)
            self._r = qt.basis(3, 2)
        else:
            raise NotImplementedError("This spin dimension is not yet implemented")

    def set_params(self, **kwargs):
        raise NotImplementedError("This is an abstract class and doesn't support this operation.")

    def get_state(self):
        raise NotImplementedError("This is an abstract class and doesn't support this operation.")

    def get_label(self, **kwargs):
        raise NotImplementedError("This is an abstract class and doesn't support this operation.")

    def _gen_ground_state_mps(self):
        """
        creates a generator for an MPS where one spin is coupled to a truncated boson space
        initialized in |0> (x) |0> for each site.
        To get an MPS you have to call this function with qtn.MatrixProductState(gen_ground_state_mps())

        :return: generator for a Matrix-product-state
        """

        phys_dim = self.spin_dim * self.dim_phonons
        excited_ind = 0
        init_bond_dim = 1

        cyc_dim = (init_bond_dim,) if self.__is_cyclic else ()
        # TODO tag all sites with their initial state-value ie. g,e,r
        f = np.zeros((*cyc_dim, init_bond_dim, phys_dim), dtype=self.__dtype)
        if self.__is_cyclic:
            f[:, :, excited_ind] = 1 / init_bond_dim
        else:
            f[:, excited_ind] = 1 / init_bond_dim
        yield f
        for j in range(self.num_spins - 2):
            s = np.zeros((init_bond_dim, init_bond_dim, phys_dim), dtype=self.__dtype)
            s[:, :, excited_ind] = 1 / init_bond_dim
            yield s
        m = np.zeros((init_bond_dim, *cyc_dim, phys_dim), dtype=self.__dtype)
        if self.__is_cyclic:
            m[:, :, excited_ind] = 1 / init_bond_dim
        else:
            m[:, excited_ind] = 1 / init_bond_dim
        yield m


class ClusterGroundState(InitialState):
    """
    Creates an initial state with all phonons in the vibrational ground state.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, cluster_size=0, excited_state=None, **kwargs):
        """
        Constructor for the ground state class.
        :param cluster_size:
        :param num_spins:
        :param dim_phonons:
        :param is_cyclic:
        :param dtype:
        :return:
        """
        super().__init__(**kwargs)
        self.cluster_size = cluster_size
        if excited_state is None:
            if self.spin_dim == 2:
                self.excited_state = self._e
            elif self.spin_dim == 3:
                self.excited_state = self._r
            else:
                raise NotImplementedError('Spin dimensions other than 2 or 3 are not implemented.')
        else:
            self.excited_state = excited_state

    def set_params(self, **kwargs):
        return

    def get_state(self):
        psi = qtn.MatrixProductState(self._gen_ground_state_mps())
        for site in range(self.cluster_size):
            nextExcitedSite = self.num_spins // 2  # set initial site to center

            # OEIS:A001057 Michael S. Branicky, Jul 14 2022
            nextExcitedSite += site // 2 + 1 if site % 2 else -site // 2
            psi.gate_(qt.tensor(self.excited_state * self._g.dag(), qt.qeye(self.dim_phonons)).data.todense(),
                      where=nextExcitedSite, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered excited Rydbergs, no phonons'


class ClusterFockState(ClusterGroundState):
    """
    Creates an initial state with phonons in the Fock state you specified.
    If no fock-states are specified, vibrational ground-state |0> is assumed.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, fock_state_array=None, **kwargs):
        super().__init__(**kwargs)
        if fock_state_array is None:
            self.fock_state_array = np.zeros([self.num_spins], dtype=int)
        elif len(fock_state_array) == self.num_spins:
            # TODO check if dtype is int
            self.fock_state_array = fock_state_array
        else:
            warnings.warn("fock_state_array didn't match the requirements. Chose zeros now.")
            self.fock_state_array = np.zeros([self.num_spins], dtype=int)

    def set_params(self, **kwargs):
        return

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            psi.gate_(qt.tensor(qt.qeye(self.spin_dim),
                                qt.basis(self.dim_phonons, self.fock_state_array[site]) *
                                qt.basis(self.dim_phonons, 0).dag()).data.todense(),
                      where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered excited Rydbergs, custom fock state'


class ClusterFockStateDualRydberg(ClusterFockState):
    """
    Creates an initial state with phonons in the Fock state you specified.
    If no fock-states are specified, vibrational ground-state |0> is assumed.
    You can specify the size of the centered initial r-Rydberg cluster.
    The other atoms are initialized in the e-state.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.spin_dim != 3:
            raise TypeError('This state works only for three-level spins.')

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            psi.gate_(qt.tensor(self._e * self._g.dag() + self._e * self._e.dag() + self._r * self._r.dag(),
                                qt.qeye(self.dim_phonons))
                      .data.todense(), where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered r excited Rydbergs, other Rydbergs in e, custom fock state'


class Cluster01State(ClusterGroundState):
    """
    Creates an initial state with phonons in the Fock state |0>+e^(i alpha)|1>.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, alpha=0, **kwargs):
        super().__init__(**kwargs)
        if self.dim_phonons < 2:
            raise AttributeError(
                f"dim_phonons={self.dim_phonons} should be at least two. Otherwise this state is meaningless.")

        self.__alpha = alpha

    def set_params(self, **kwargs):
        """
        sets the alpha parameter
        :key alpha: phase factor in the 01 state
        """
        try:
            self.__alpha = kwargs['alpha']
        except:
            warnings.warn(f'No alpha specified for 01-state. Chose alpha={self.__alpha}.')
        return

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            psi.gate_(qt.tensor(qt.qeye(self.spin_dim),
                                (qt.basis(self.dim_phonons, 0) + np.exp(1j * self.__alpha)
                                 * qt.basis(self.dim_phonons, 1)).unit()
                                * qt.basis(self.dim_phonons, 0).dag()).data.todense(),
                      where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered excited Rydbergs, 01 fock state, alpha={self.__alpha}'


class ClusterCoherentState(ClusterGroundState):
    """
    Creates an initial state with phonons in the coherent fock state.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, alpha=0, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def set_params(self, **kwargs):
        try:
            self._alpha = kwargs['alpha']
        except:
            warnings.warn(f'No alpha specified for coherent-state. Chose alpha={self._alpha}.')
        return

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            psi.gate_(
                qt.tensor(qt.qeye(self.spin_dim), qt.coherent(self.dim_phonons, alpha=self._alpha)
                          * qt.basis(self.dim_phonons, 0).dag()).data.todense(),
                where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered excited Rydbergs, coherent fock state, alpha={self._alpha}'


class ClusterCoherentAndersonState(ClusterGroundState):
    """
    Creates an initial state with phonons in the coherent fock state.
    The initial phase is chosen random.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, alpha=0, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha
        self._rng = np.random.default_rng()

    def set_params(self, **kwargs):
        try:
            self._alpha = kwargs['alpha']
        except:
            warnings.warn(f'No alpha specified for coherent-state. Chose alpha={self._alpha}.')
        return

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            rand_phase = self._rng.uniform(0, 2 * np.pi)
            psi.gate_(
                qt.tensor(qt.qeye(self.spin_dim),
                          qt.coherent(self.dim_phonons, alpha=self._alpha * np.exp(1j * rand_phase))
                          * qt.basis(self.dim_phonons, 0).dag()).data.todense(),
                where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return rf'{self.cluster_size} initial centered excited Rydbergs, coherent anderson fock state, alpha={self._alpha}'


class ClusterQuantumTemperature(ClusterGroundState):
    """
    Creates an initial state with phonons in the coherent fock state.
    You can specify the size of the centered initial Rydberg cluster.
    """

    def __init__(self, beta=0, omega=0, **kwargs):
        super().__init__(**kwargs)
        self.__beta = beta
        self.__omega = omega

    def set_params(self, **kwargs):
        try:
            self.__beta = kwargs['beta']
            self.__omega = kwargs['omega']
        except:
            warnings.warn(
                f'No beta or omega specified for coherent-state.'
                f'Chose beta={self.__beta},omega={self.__omega}.')
        return

    def get_state(self):
        psi = super().get_state()
        for site in range(self.num_spins):
            quantum_thermal = 0 * qt.basis(self.dim_phonons)
            for num_of_chunks in range(self.dim_phonons):
                # TODO check: we normalize here after summing up all states
                quantum_thermal += np.exp(-self.__beta * self.__omega / 2 * num_of_chunks) * qt.basis(self.dim_phonons,
                                                                                                      num_of_chunks)

            psi.gate_(
                qt.tensor(qt.qeye(self.spin_dim),
                          quantum_thermal.unit() * qt.basis(self.dim_phonons, 0).dag()).data.todense(),
                where=site, contract=True)
        return psi

    def get_label(self, **kwargs):
        return f'{self.cluster_size} initial centered excited Rydbergs, \
        quantum thermal state, beta={self.__beta}, omega={self.__omega}'


def hamiltonian_2_level_spin(num_spins=2, dim_phonons=1, is_cyclic=False, **kwargs):
    """
    Creates the 2-level Hamiltonian with given parameters.

    :keyword OmegaE: Rabi frequency between |g> and |e>
    :keyword V_e_vdw: nearest neighbour interaction strength in states |e> <-> |e>
    :keyword omega: trap frequency
    :keyword kappa_e: linear taylor expansion of the potential between |e> <-> |e>
    :return: LocalHam1D
    """
    spin_dim = 2
    g = qt.basis(spin_dim, 0)
    e = qt.basis(spin_dim, 1)

    # TODO change positional arguments to named parameters
    # construct hamiltonian between nearest neighbours (4M**2 x 4M**2)
    EE = qt.tensor(e * e.dag(), qt.qeye(dim_phonons))
    multi_ee = multi_spin_operator(EE, 2)

    linearIntOp = linear_int_operator(e * e.dag(), dim_phonons=dim_phonons)

    DeltaEE = -kwargs['V_e_vdw']
    # construct hamiltonian for one atom (2x2)
    H_OmegaE = kwargs['OmegaE'] * qt.tensor(e * g.dag() + g * e.dag(), qt.qeye(dim_phonons))
    H_Delta = DeltaEE * qt.tensor(e * e.dag(), qt.qeye(dim_phonons))
    H_phonon = kwargs['omega'] * qt.tensor(qt.qeye(spin_dim), qt.num(dim_phonons) + 0.5)

    H_singleSite = H_OmegaE + H_Delta + H_phonon

    # We don't have to divide the interaction strength by the number of neighbours as in the QuTiP-simulation!
    # QUIMB handles this already
    H_vdw = kwargs['V_e_vdw'] * multi_ee[0] * multi_ee[1]
    # TODO implement realistic kappa from potential derivative
    H_vdw -= kwargs['kappa_e'] * linearIntOp

    H_neighbourSite = np.array(H_vdw.data.todense())

    return qtn.LocalHam1D(L=num_spins, H2=H_neighbourSite, H1=np.array(H_singleSite.data.todense()),
                          cyclic=is_cyclic)


def hamiltonian_3_level_dual_species(num_spins=2, dim_phonons=1, is_cyclic=False, **kwargs):
    """
    Create the dual species interaction Hamiltonian for three spins.

    :param num_spins: Number of spins in the 1D chain
    :param dim_phonons: dimension of phonon Hilbert space
    :param is_cyclic: True if is_cyclic boundaries
    :return:
    """
    spin_dim = 3

    g = qt.basis(spin_dim, 0)
    e = qt.basis(spin_dim, 1)
    r = qt.basis(spin_dim, 2)

    multiEE = multi_spin_operator(qt.tensor(e * e.dag(), qt.qeye(dim_phonons)), 2)
    multiRR = multi_spin_operator(qt.tensor(r * r.dag(), qt.qeye(dim_phonons)), 2)
    multiRe = multi_spin_operator(qt.tensor(r * e.dag(), qt.qeye(dim_phonons)), 2)
    multiEr = multi_spin_operator(qt.tensor(e * r.dag(), qt.qeye(dim_phonons)), 2)

    phonon_x = multi_spin_operator(qt.tensor(qt.qeye(spin_dim), qt.position(dim_phonons)), 2)

    # construct hamiltonian for one atom (3x3)
    H_OmegaE = kwargs['OmegaE'] * qt.tensor(e * g.dag() + g * e.dag(), qt.qeye(dim_phonons))
    H_delta_e = kwargs['DeltaEE'] * qt.tensor(e * e.dag(), qt.qeye(dim_phonons))
    H_delta_r = (kwargs['DeltaEE'] + kwargs['omegaR']) * qt.tensor(r * r.dag(), qt.qeye(dim_phonons))
    H_phonon = kwargs['omega'] * qt.tensor(qt.qeye(spin_dim), qt.num(dim_phonons) + 0.5)

    H_singleSite = H_OmegaE + H_delta_e + H_delta_r + H_phonon

    # We don't have to divide the interaction strength by the number of neighbours as in the QuTiP-simulation!
    # QUIMB handles this already
    H_vdw_e = multiEE[0] * multiEE[1] * (kwargs['V_e_vdw'] + kwargs['kappa_e'] * (phonon_x[0] - phonon_x[1]))
    H_vdw_r = multiRR[0] * multiRR[1] * (kwargs['V_r_vdw'] + kwargs['kappa_r'] * (phonon_x[0] - phonon_x[1]))
    H_dd = (multiRe[0] * multiEr[1] + multiEr[0] * multiRe[1]) * (
            kwargs['V_dd'] + kwargs['kappa_dd'] * (phonon_x[0] - phonon_x[1]))

    H_neighbourSite = H_vdw_e + H_vdw_r + H_dd

    return qtn.LocalHam1D(
        L=num_spins,
        H2=np.array(H_neighbourSite.data.todense()),
        H1=np.array(H_singleSite.data.todense()),
        cyclic=is_cyclic)


def hamiltonian_3_level_vee(num_spins=2, dim_phonons=1, is_cyclic=False, **kwargs):
    """
    Create the Vee Hamiltonian for three spins.

    :param num_spins: Number of spins in the 1D chain
    :param dim_phonons: dimension of phonon Hilbert space
    :param is_cyclic: True if is_cyclic boundaries
    :return:
    """
    spin_dim = 3

    g = qt.basis(spin_dim, 0)
    e = qt.basis(spin_dim, 1)
    r = qt.basis(spin_dim, 2)

    multiEE = multi_spin_operator(qt.tensor(e * e.dag(), qt.qeye(dim_phonons)), 2)
    multiRR = multi_spin_operator(qt.tensor(r * r.dag(), qt.qeye(dim_phonons)), 2)
    multiRe = multi_spin_operator(qt.tensor(r * e.dag(), qt.qeye(dim_phonons)), 2)
    multiEr = multi_spin_operator(qt.tensor(e * r.dag(), qt.qeye(dim_phonons)), 2)

    phonon_x = multi_spin_operator(qt.tensor(qt.qeye(spin_dim), qt.position(dim_phonons)), 2)

    # construct hamiltonian for one atom (3x3)
    H_OmegaE = kwargs['OmegaE'] * qt.tensor(e * g.dag() + g * e.dag(), qt.qeye(dim_phonons))
    H_OmegaR = kwargs['OmegaR'] * qt.tensor(r * g.dag() + g * r.dag(), qt.qeye(dim_phonons))
    H_delta_e = kwargs['DeltaEE'] * qt.tensor(e * e.dag(), qt.qeye(dim_phonons))
    H_delta_r = kwargs['DeltaRR'] * qt.tensor(r * r.dag(), qt.qeye(dim_phonons))
    H_phonon = kwargs['omega'] * qt.tensor(qt.qeye(spin_dim), qt.num(dim_phonons) + 0.5)

    H_singleSite = H_OmegaE + H_OmegaR + H_delta_e + H_delta_r + H_phonon

    # We don't have to divide the interaction strength by the number of neighbours as in the QuTiP-simulation!
    # QUIMB handles this already
    H_vdw_e = multiEE[0] * multiEE[1] * (kwargs['V_e_vdw'] + kwargs['kappa_e'] * (phonon_x[0] - phonon_x[1]))
    H_vdw_r = multiRR[0] * multiRR[1] * (kwargs['V_r_vdw'] + kwargs['kappa_r'] * (phonon_x[0] - phonon_x[1]))
    H_dd = (multiRe[0] * multiEr[1] + multiEr[0] * multiRe[1]) * (
            kwargs['V_dd'] + kwargs['kappa_dd'] * (phonon_x[0] - phonon_x[1]))

    H_neighbourSite = H_vdw_e + H_vdw_r + H_dd

    return qtn.LocalHam1D(
        L=num_spins,
        H2=np.array(H_neighbourSite.data.todense()),
        H1=np.array(H_singleSite.data.todense()),
        cyclic=is_cyclic)


# %%##################################################################################################
# SPECIFIC QUANTUM HELPER FUNCTIONS
######################################################################################################
def multi_spin_operator(op, num_of_particles):
    """
    creates the many-body single-site operators
    where first list entry corresponds to op on first site, qeye on all others

    :param op: single-site operator
    :param num_of_particles: number of particles
    :return: list of single-site operators
    """
    # TODO implement next nearest neighbour interaction
    dim = op.dims[0]
    ret_op = []
    for i in range(num_of_particles):
        if i == 0:
            ret_op.append(op)
        else:
            ret_op.append(qt.qeye(dim))
        for j in range(num_of_particles - 1):
            if j + 1 == i:
                ret_op[i] = qt.tensor(ret_op[i], op)
            else:
                ret_op[i] = qt.tensor(ret_op[i], qt.qeye(dim))
    return ret_op


def linear_int_operator(spin_state, operates='single', dim_phonons=1):
    """
    Calculates the phonon interaction operator in linear order
    :param dim_phonons: dimension of phonon Hilbert space
    :param spin_state: a single or dual state projector from qutip. I.e. e*e.dag()
    :param operates: Only single is supported for now
    :return:
    """
    a = qt.destroy(dim_phonons)

    if operates == 'single':
        SS = qt.tensor(spin_state, qt.qeye(dim_phonons))
    else:
        raise NotImplementedError('operates parameter is not implemented.')

    multi_SS = multi_spin_operator(SS, 2)
    multi_SS_apD = multi_spin_operator(qt.tensor(spin_state, a + a.dag()), 2)
    linear_int_op_SS = multi_SS_apD[0] * multi_SS[1] - multi_SS_apD[1] * multi_SS[0]
    return linear_int_op_SS


def expect_one_site(j, psi=None, observables=None, num_spins=1, dtype='complex128', is_cyclic=False):
    """
    Calculates the expectation values specified above for given state psi at site j.
    Helper-function for parallel processing

    :param j: site to calculate observables for
    :param psi: current MPS-state
    :param observables: list of observables to measure at site j with entry `single-site` and `other`
    :param num_spins: Number of spins in the MPS
    :param dtype: datatype of the observables
    :param is_cyclic: True if boundary conditions of MPS are is_cyclic
    :return: numpy array of single site observables
    """

    if psi is None:
        raise AttributeError('You have to specify a state psi.')
    if observables is None:
        raise AttributeError('You have to specify a list of observables with entry `single-site` and `other`.')

    n_of_ss_e_ops = len(observables['single_site'])
    expectation_values = np.empty(n_of_ss_e_ops + 2, dtype=dtype)
    # TODO refactor this later and use the op_name instead of numerical index
    i = 0
    for op_name, op in observables['single_site'].items():
        expectation_values[i] = psi.H @ psi.gate(op, j)
        i += 1

    expectation_values[n_of_ss_e_ops] = np.amax(psi[j].shape)  # corresponds to max bond dim of this site

    # bipartite entropy of the left block with j sites, left and right boundary is undefined
    if j == 0 or j == num_spins - 1 or is_cyclic:
        expectation_values[n_of_ss_e_ops + 1] = -1
    else:
        expectation_values[n_of_ss_e_ops + 1] = psi.entropy(j)
    # see https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html#quimb.tensor.tensor_1d.MatrixProductState.entropy

    return expectation_values


# %%##################################################################################################
# GENERAL UTILITIES
######################################################################################################
def flatten(lst: list) -> list:
    """
    Utility function to flatten a nested list.
    
    :param lst: list
    :return: flattened list
    """
    return [item for sublist in lst for item in sublist]


def split_in_chunks(lst: list, num_of_chunks: int) -> list:
    """
    Splits a list into num_of_chunks chunks.

    :param lst: list to split
    :param num_of_chunks: number of chunks
    :return: list of num_of_chunks lists
    """
    return [lst[i::num_of_chunks] for i in range(num_of_chunks)]


def dict_product(dicts: dict) -> list:
    """
    Returns the cartesian product of a dictionary of lists.

    :param dicts: dictionary of lists
    :return: list of dictionaries
    """
    return [dict(zip(dicts.keys(), x)) for x in product(*dicts.values())]
