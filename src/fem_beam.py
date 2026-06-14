"""
fem_beam
========

A finite element solver for Euler-Bernoulli beam bending and vibration,
implemented in NumPy/SciPy.

The beam is discretised with 2-node cubic (Hermite) elements. Each node carries
two degrees of freedom: transverse deflection ``w`` and rotation ``theta``.
The element stiffness and consistent-mass matrices are the standard closed-form
4x4 matrices derived from the cubic shape functions:

    Ke = (EI / L^3) *  [[ 12,   6L,  -12,   6L ],
                        [ 6L, 4L^2,  -6L, 2L^2 ],
                        [-12,  -6L,   12,  -6L ],
                        [ 6L, 2L^2,  -6L, 4L^2 ]]

    Me = (rhoA L / 420) * [[ 156,   22L,    54,  -13L ],
                           [ 22L,  4L^2,   13L, -3L^2 ],
                           [  54,   13L,   156,  -22L ],
                           [-13L, -3L^2,  -22L,  4L^2 ]]

Supported analyses:
  * static bending          K u = f
  * modal / free vibration  K phi = omega^2 M phi
  * transient dynamics      M a + C v + K u = f(t)   (Newmark-beta)

The section can be uniform or vary element-by-element (via ``EI_profile`` /
``rhoA_profile``), which captures stepped or cut-out beams. Validated against
analytical solutions in tests/.
"""
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from scipy.linalg import eigh

# Dimensionless eigenvalues (beta_n * L) for a clamped-free Euler-Bernoulli beam
CANTILEVER_BETA_L = np.array([1.8751040687, 4.6940911330, 7.8547574382,
                              10.9955407349, 14.1371683910])


@dataclass
class Beam:
    """A clamped-free Euler-Bernoulli beam.

    Parameters
    ----------
    length : float        beam length L [m]
    E : float             Young's modulus [Pa]
    I : float             second moment of area [m^4]
    rho : float           density [kg/m^3]
    A : float             cross-sectional area [m^2]
    n_elements : int      number of finite elements
    EI_profile : array    optional per-element bending stiffness E*I (len n_elements)
    rhoA_profile : array  optional per-element mass-per-length rho*A (len n_elements)
    """
    length: float
    E: float
    I: float
    rho: float
    A: float
    n_elements: int = 20
    EI_profile: Optional[np.ndarray] = None
    rhoA_profile: Optional[np.ndarray] = None

    # --- geometry helpers ---
    @property
    def n_nodes(self):
        return self.n_elements + 1

    @property
    def n_dof(self):
        return 2 * self.n_nodes

    @property
    def le(self):
        return self.length / self.n_elements

    @property
    def node_x(self):
        return np.linspace(0.0, self.length, self.n_nodes)

    def _EI_e(self, e):
        if self.EI_profile is not None:
            return float(self.EI_profile[e])
        return self.E * self.I

    def _rhoA_e(self, e):
        if self.rhoA_profile is not None:
            return float(self.rhoA_profile[e])
        return self.rho * self.A

    # --- element matrices ---
    def _element_stiffness(self, EI):
        L = self.le
        return (EI / L ** 3) * np.array([
            [12,    6 * L,   -12,    6 * L],
            [6 * L, 4 * L * L, -6 * L, 2 * L * L],
            [-12,  -6 * L,    12,   -6 * L],
            [6 * L, 2 * L * L, -6 * L, 4 * L * L],
        ])

    def _element_mass(self, rhoA):
        L = self.le
        return (rhoA * L / 420.0) * np.array([
            [156,    22 * L,    54,   -13 * L],
            [22 * L, 4 * L * L, 13 * L, -3 * L * L],
            [54,     13 * L,   156,   -22 * L],
            [-13 * L, -3 * L * L, -22 * L, 4 * L * L],
        ])

    # --- assembly ---
    def assemble(self):
        """Assemble the global stiffness K and consistent mass M matrices."""
        n = self.n_dof
        K = np.zeros((n, n))
        M = np.zeros((n, n))
        for e in range(self.n_elements):
            dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(dofs, dofs)] += self._element_stiffness(self._EI_e(e))
            M[np.ix_(dofs, dofs)] += self._element_mass(self._rhoA_e(e))
        return K, M

    def _free_dofs(self):
        # Clamp node 0: remove w0 (dof 0) and theta0 (dof 1).
        return np.arange(2, self.n_dof)

    def _load_vector(self, tip_load=0.0, distributed_load=0.0):
        f = np.zeros(self.n_dof)
        if distributed_load:
            L = self.le
            q = distributed_load
            fe = np.array([q * L / 2, q * L * L / 12, q * L / 2, -q * L * L / 12])
            for e in range(self.n_elements):
                dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
                f[dofs] += fe
        if tip_load:
            f[2 * (self.n_nodes - 1)] += tip_load
        return f

    # --- static analysis ---
    def _static_full(self, tip_load=0.0, distributed_load=0.0):
        K, _ = self.assemble()
        f = self._load_vector(tip_load, distributed_load)
        free = self._free_dofs()
        u = np.zeros(self.n_dof)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        return u

    def static_deflection(self, tip_load=0.0, distributed_load=0.0):
        """Solve static bending; return node x-coordinates and deflection w."""
        u = self._static_full(tip_load, distributed_load)
        return self.node_x, u[0::2]

    # --- modal analysis ---
    def natural_frequencies(self, n_modes=4):
        """Lowest ``n_modes`` natural frequencies [Hz] and mode shapes (nodal w)."""
        K, M = self.assemble()
        free = self._free_dofs()
        eigvals, eigvecs = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
        eigvals = np.clip(eigvals, 0.0, None)
        freqs = np.sqrt(eigvals) / (2 * np.pi)

        shapes = []
        for k in range(min(n_modes, len(freqs))):
            full = np.zeros(self.n_dof)
            full[free] = eigvecs[:, k]
            shape = full[0::2]
            if shape[np.argmax(np.abs(shape))] < 0:
                shape = -shape
            shapes.append(shape)
        return freqs[:n_modes], np.array(shapes)

    # --- transient dynamics (Newmark-beta) ---
    def transient_response(self, t_final, n_steps, preload_tip=0.0,
                           preload_distributed=0.0, force=None,
                           rayleigh=(0.0, 0.0), beta=0.25, gamma=0.5):
        """Integrate  M a + C v + K u = f(t)  with the Newmark-beta method.

        Parameters
        ----------
        t_final : float            total simulated time [s]
        n_steps : int              number of time steps
        preload_tip,
        preload_distributed        static load whose deflection is used as the
                                   initial condition u0 (then released unless a
                                   ``force`` keeps acting). Models a beam pulled
                                   to a static shape and let go (ring-down).
        force : callable(t)->vec   optional time-varying global force vector
                                   (length n_dof). Default: free vibration.
        rayleigh : (alpha_m, alpha_k)  Rayleigh damping  C = am*M + ak*K
        beta, gamma : float        Newmark parameters (default = average
                                   acceleration: unconditionally stable).

        Returns
        -------
        t : ndarray            time stations (n_steps+1,)
        tip_w : ndarray        tip deflection history
        u_hist : ndarray       full nodal-deflection history (n_steps+1, n_nodes)
        """
        K, M = self.assemble()
        free = self._free_dofs()
        Kr = K[np.ix_(free, free)]
        Mr = M[np.ix_(free, free)]
        am, ak = rayleigh
        Cr = am * Mr + ak * Kr

        dt = t_final / n_steps
        nf = len(free)

        # initial conditions
        u0 = np.zeros(self.n_dof)
        if preload_tip or preload_distributed:
            u0 = self._static_full(preload_tip, preload_distributed)
        u = u0[free].copy()
        v = np.zeros(nf)

        def fvec(t):
            return force(t)[free] if force is not None else np.zeros(nf)

        a = np.linalg.solve(Mr, fvec(0.0) - Cr @ v - Kr @ u)

        # Newmark constants
        a0 = 1.0 / (beta * dt ** 2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2 * beta) - 1.0)

        Keff = Kr + a0 * Mr + a1 * Cr
        # factorise once (constant dt, linear system)
        from scipy.linalg import lu_factor, lu_solve
        lu = lu_factor(Keff)

        tip_dof_local = np.where(free == 2 * (self.n_nodes - 1))[0][0]
        n_out = n_steps + 1
        tip_w = np.zeros(n_out)
        u_hist = np.zeros((n_out, self.n_nodes))
        t = np.linspace(0.0, t_final, n_out)

        full = np.zeros(self.n_dof)
        full[free] = u
        tip_w[0] = u[tip_dof_local]
        u_hist[0] = full[0::2]

        for n in range(1, n_out):
            tn = t[n]
            rhs = (fvec(tn)
                   + Mr @ (a0 * u + a2 * v + a3 * a)
                   + Cr @ (a1 * u + a4 * v + a5 * a))
            u_new = lu_solve(lu, rhs)
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1 - gamma) * a + gamma * a_new)
            u, v, a = u_new, v_new, a_new

            full = np.zeros(self.n_dof)
            full[free] = u
            tip_w[n] = u[tip_dof_local]
            u_hist[n] = full[0::2]

        return t, tip_w, u_hist

    # --- analytical references (for validation/plots) ---
    def analytical_tip_deflection(self, tip_load=0.0, distributed_load=0.0):
        EI = self.E * self.I
        L = self.length
        w = 0.0
        if tip_load:
            w += tip_load * L ** 3 / (3 * EI)
        if distributed_load:
            w += distributed_load * L ** 4 / (8 * EI)
        return w

    def analytical_frequencies(self, n_modes=4):
        EI = self.E * self.I
        rhoA = self.rho * self.A
        L = self.length
        bl = CANTILEVER_BETA_L[:n_modes]
        omega = bl ** 2 * np.sqrt(EI / (rhoA * L ** 4))
        return omega / (2 * np.pi)


def steel_rectangular(length=1.0, width=0.05, height=0.01, n_elements=20):
    """Convenience constructor: a steel beam with a rectangular cross-section."""
    A = width * height
    I = width * height ** 3 / 12.0
    return Beam(length=length, E=200e9, I=I, rho=7850.0, A=A, n_elements=n_elements)


# Common engineering materials: (Young's modulus [Pa], density [kg/m^3])
MATERIALS = {
    "steel":    (200e9, 7850.0),
    "aluminum": (69e9,  2710.0),
    "copper":   (117e9, 8960.0),
    "titanium": (116e9, 4506.0),
}


def beam_for_material(material, length=1.0, width=0.05, height=0.01, n_elements=20):
    """Build a rectangular beam for a named material from ``MATERIALS``."""
    E, rho = MATERIALS[material]
    A = width * height
    I = width * height ** 3 / 12.0
    return Beam(length=length, E=E, I=I, rho=rho, A=A, n_elements=n_elements)


def rectangular_section(width, height):
    """Return (A, I) for a solid rectangular cross-section."""
    return width * height, width * height ** 3 / 12.0
