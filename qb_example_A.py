"""
Example A of Yang & del Campo, "Minimum-Time Quantum Control and the Quantum
Brachistochrone Equation" (arXiv:2204.12792): one qubit, restricted operator X1 = sigma_z.

Closed-subalgebra solution (Eqs. 87-88, with lambda_0 = 1, mu_x = 0, mu_y = omega):
    H(t) = omega * exp(i l1 Z t) Y exp(-i l1 Z t)
    U(t) = exp(i l1 Z t) * exp(-i (omega Y + l1 Z) t)
    F(t) = U(t) F(0) U(t)^dagger,   F(0) = omega Y + l1 Z
Initial state |psi_i> = |+x>.  Final state |psi_f> = U(T)|psi_i> unless stated.

The script
  1. builds U(T) as a Qiskit circuit and checks it against the analytic matrix;
  2. checks that H(t) = i dU/dt U^dagger and that F obeys the CHKO equation;
  3. checks the extra endpoint condition (Eq. 55 / Eq. 86): Im<psi_f|H(T)F(T)|psi_f> ~ lambda1*cos(2*Lambda1*T);
  4. reproduces Fig. 2 (a, b) of the paper;
  5. (see NOTE at the end of this docstring about Eq. 55)
     for a prescribed target |psi_f(phi, Omega_B)> evaluates the closed forms (98)-(105): the
     compatibility condition (105) picks the allowed (k, l), Eq. (103) gives T(k, l), and the
     result is verified with a Qiskit circuit.  NOTE: (phi, Omega_B) must lie on a curve
     satisfying Eq. (105); a generic target is NOT reached by an extremal trajectory of this form
     (3 conditions - 2 for the state, 1 for Eq. 55 - on 2 unknowns lambda1, T).

NOTE on Eq. (55): in my numerical tests (section 6) the true minimum-time trajectory for a
generic target violates Eq. (55) of the paper (Im<psi_f|H F|psi_f> != 0), while still being an
Example-A trajectory (H(0) = -omega*sigma_y branch).  Sections 1-5 reproduce the paper as written;
section 6, `quantum_brachistochrone`, does not use Eq. (55) as a filter.

Requires: qiskit, numpy, scipy, matplotlib.   Run:  python qb_example_A.py
"""
import itertools
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

warnings.filterwarnings("ignore", message=".*(splu|spsolve).*")  # scipy sparse noise from PauliEvolutionGate

OMEGA = 10.0  # norm bound, as in Fig. 2 of the paper

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PSI_I = np.array([1, 1], dtype=complex) / np.sqrt(2)  # |+x>


# ----------------------------------------------------------------------------
# Analytic solution (Eqs. 87-88)
# ----------------------------------------------------------------------------
def H_t(t, l1, w=OMEGA):
    R = expm(1j * l1 * Z * t)
    return w * R @ Y @ R.conj().T


def U_t(t, l1, w=OMEGA):
    return expm(1j * l1 * Z * t) @ expm(-1j * (w * Y + l1 * Z) * t)


def F_t(t, l1, w=OMEGA):
    U = U_t(t, l1, w)
    return U @ (w * Y + l1 * Z) @ U.conj().T


# ----------------------------------------------------------------------------
# Qiskit circuit for U(T)|+x>
# ----------------------------------------------------------------------------
def qb_circuit(T, l1, w=OMEGA):
    """U(T) = exp(i l1 Z T) exp(-i (w Y + l1 Z) T), applied to |+x> = H|0>.

    Circuit order (time order): H, exp(-i(wY + l1 Z)T), then exp(+i l1 Z T) = RZ(-2 l1 T).
    """
    qc = QuantumCircuit(1)
    qc.h(0)  # |0> -> |+x>
    generator = SparsePauliOp.from_list([("Y", w), ("Z", l1)])
    qc.append(PauliEvolutionGate(generator, time=T), [0])  # exp(-i (wY+l1 Z) T)
    qc.rz(-2 * l1 * T, 0)  # exp(+i l1 Z T)
    return qc


# ----------------------------------------------------------------------------
# Endpoint quantities
# ----------------------------------------------------------------------------
def endpoint_quantities(T, l1, w=OMEGA):
    """Return (|<psi_f|psi_i>|, c, Im<psi_f|H F~|psi_f>) with psi_f = U(T) psi_i
    and F~ = F / c, c = Re<psi_f|H(T)F(T)|psi_f>  (the rescaling that enforces Eq. 54)."""
    U = U_t(T, l1, w)
    psi_f = U @ PSI_I
    HF = H_t(T, l1, w) @ F_t(T, l1, w)
    val = psi_f.conj() @ HF @ psi_f
    c = val.real
    return abs(psi_f.conj() @ PSI_I), c, (val / c).imag


# ----------------------------------------------------------------------------
# 1-3. Checks
# ----------------------------------------------------------------------------
def run_checks():
    print("=== Checks (omega = %g) ===" % OMEGA)
    T, l1 = 0.21, 3.7

    # circuit vs analytic
    sv = Statevector(qb_circuit(T, l1))
    psi = U_t(T, l1) @ PSI_I
    fid = abs(np.vdot(psi, sv.data)) ** 2
    print(f"circuit vs analytic fidelity           : {fid:.12f}")

    # H = i dU/dt U^dagger  (central difference)
    dt = 1e-6
    dU = (U_t(T + dt, l1) - U_t(T - dt, l1)) / (2 * dt)
    H_num = 1j * dU @ U_t(T, l1).conj().T
    print(f"|H(t) - i U' U^dag|                    : {np.abs(H_num - H_t(T, l1)).max():.2e}")

    # constraints Tr H = 0, (1/2)Tr H^2 = omega^2, Tr(H Z) = 0
    H = H_t(T, l1)
    print(f"Tr H                                   : {np.trace(H).real:.2e}")
    print(f"(1/2)Tr H^2 - omega^2                  : {0.5*np.trace(H@H).real - OMEGA**2:.2e}")
    print(f"Tr(H sigma_z)                          : {np.trace(H@Z).real:.2e}")

    # CHKO:  F' + i[H, F] = 0
    dF = (F_t(T + dt, l1) - F_t(T - dt, l1)) / (2 * dt)
    F = F_t(T, l1)
    print(f"|F' + i[H,F]|                          : {np.abs(dF + 1j*(H@F - F@H)).max():.2e}")

    # initial condition (14): {F(0), P(0)} = F(0)
    P0 = np.outer(PSI_I, PSI_I.conj())
    F0 = F_t(0, l1)
    print(f"|{{F(0),P(0)}} - F(0)|                   : {np.abs(F0@P0 + P0@F0 - F0).max():.2e}")

    # Eq. (96): the endpoint condition is proportional to l1 cos(2 Lambda1 T)
    print("\nEq. (55)/(86) vs closed form  l1*cos(2*Lambda1*T):")
    for (T_, l_) in [(0.13, 2.0), (0.21, 3.7), (0.30, 9.0)]:
        Lam = np.hypot(l_, OMEGA)
        _, c, im = endpoint_quantities(T_, l_)
        # Eq. (86): <psi_i|[H(0)+G, U^dag G U]|psi_i> with lambda_0 = 1, G = l1 Z
        U = U_t(T_, l_)
        A = OMEGA * Y + l_ * Z
        B = U.conj().T @ (l_ * Z) @ U
        eq86 = PSI_I.conj() @ (A @ B - B @ A) @ PSI_I
        print(f"  T={T_:.2f} l1={l_:4.1f}:  Im<HF~>={im: .6f}   Eq.86={eq86.imag: .6f}"
              f"   l1*cos(2*Lam*T)={l_*np.cos(2*Lam*T_): .6f}   c={c:.4f}")
    print()


# ----------------------------------------------------------------------------
# 4. Fig. 2 of the paper
# ----------------------------------------------------------------------------
def make_figure(path="qb_fig2.png", n_l=161, n_T=161):
    # axes of the paper: omega*T/pi in (0.2, 1.0],  lambda~1 = lambda1/c in (0, 0.2]
    wT = np.linspace(0.2, 1.0, n_T)
    lt = np.linspace(0.001, 0.2, n_l)
    Ts = wT * np.pi / OMEGA
    amp = np.zeros((n_l, n_T))
    imag = np.zeros((n_l, n_T))
    for a, lam_tilde in enumerate(lt):
        for b, T in enumerate(Ts):
            # lambda~1 = lambda1 / c.  c itself depends on lambda1, so solve by fixed point.
            l1 = lam_tilde * OMEGA**2  # c = omega^2 + corrections
            for _ in range(3):
                _, c, _ = endpoint_quantities(T, l1)
                l1 = lam_tilde * c
            amp[a, b], _, imag[a, b] = endpoint_quantities(T, l1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    p0 = axs[0].contourf(wT, lt, amp, levels=20, cmap="YlOrBr")
    axs[0].set_title(r"(a)  $|\langle\psi_f|\psi_i\rangle|$")
    p1 = axs[1].contourf(wT, lt, imag, levels=np.linspace(-2, 2, 21), cmap="viridis", extend="both")
    axs[1].contour(wT, lt, imag, levels=[0], colors="k", linewidths=1.2)
    axs[1].set_title(r"(b)  Im$\langle\psi_f|H(T)\tilde F(T)|\psi_f\rangle$  (black: Im = 0)")
    for ax in axs:
        ax.set_xlabel(r"$\omega T/\pi$")
    axs[0].set_ylabel(r"$\tilde\lambda_1$")
    fig.colorbar(p0, ax=axs[0])
    fig.colorbar(p1, ax=axs[1])
    fig.suptitle(rf"Example A, $\omega={OMEGA:g}$: black contours = pairs satisfying the paper's Eq. (55)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


# ----------------------------------------------------------------------------
# 5. Prescribed target: Eqs. (97)-(105)
# ----------------------------------------------------------------------------
MINUS_X = np.array([1, -1], dtype=complex) / np.sqrt(2)


def target_state(phi, OmB):
    """Eq. (97) / (65):  cos(OmB) e^{i phi} |+x> + sin(OmB) |-x>  (computational basis)."""
    return np.array([np.exp(1j * phi) * np.cos(OmB) + np.sin(OmB),
                     np.exp(1j * phi) * np.cos(OmB) - np.sin(OmB)]) / np.sqrt(2)


def target_params(psi):
    """(phi, Omega_B) of a state, with the global phase fixed so <-x|psi> is real and >= 0."""
    psi = psi * np.exp(-1j * np.angle(np.vdot(MINUS_X, psi)))
    a, b = np.vdot(PSI_I, psi), np.vdot(MINUS_X, psi).real
    return np.angle(a), np.arctan2(b, abs(a))


def arccot(x):
    """Principal arccot with values in (0, pi); other branches are absorbed by the integer l."""
    return np.arctan2(1.0, x)


def solve_known_target(phi, OmB, w=OMEGA, kmax=6, lmax=6, tol=1e-6):
    """Closed-form candidates T(k, l) of Eq. (103) that satisfy the compatibility condition
    Eq. (105).  Each is then confirmed against the real dynamics (fidelity and Eq. 55)."""
    psi_f = target_state(phi, OmB)
    ac = arccot(np.sin(phi) * np.tan(2 * OmB))
    lhs_fac = np.sqrt(1 + (1 / np.tan(2 * OmB)) ** 2 / np.sin(phi) ** 2)
    rhs = np.sin(2 * OmB) * np.sin(phi)
    out = []
    for k, l in itertools.product(range(0, kmax + 1), range(-lmax, lmax + 1)):
        disc = (np.pi / 4 + k * np.pi / 2) ** 2 - 0.25 * (ac + l * np.pi) ** 2
        if disc <= 0:
            continue
        cond105 = 2 * (ac / np.pi + l) / ((1 + 2 * k) * lhs_fac)  # lhs of Eq. (105)
        if abs(cond105 - rhs) > tol:
            continue
        T = np.sqrt(disc) / abs(w)  # Eq. (103)
        l1 = 0.5 * (ac + l * np.pi) / T  # Eq. (102)
        fid = abs(np.vdot(psi_f, U_t(T, l1, w) @ PSI_I)) ** 2
        _, _, im = endpoint_quantities(T, l1, w)
        out.append((T, k, l, l1, fid, im))
    return sorted(out)


def demo_known_target():
    print("=== Prescribed target (Eqs. 97-105) ===")
    # A target that lies on an extremal curve: take the k=0, lambda1=7 trajectory.
    k0, l1_0 = 0, 7.0
    T0 = (np.pi / 4 + k0 * np.pi / 2) / np.hypot(l1_0, OMEGA)
    phi, OmB = target_params(U_t(T0, l1_0) @ PSI_I)
    print(f"target from (k=0, lambda1={l1_0}): phi = {phi:.5f}, Omega_B = {OmB:.5f}")
    print(f"unrestricted minimum time Omega_B/omega = {OmB / OMEGA:.6f}")

    sols = solve_known_target(phi, OmB)
    if not sols:
        print("no (k, l) satisfies Eq. (105) for this target")
        return
    print("\n   T            k   l   lambda1      fidelity      Im<HF~>")
    for T, k, l, l1, fid, im in sols[:6]:
        print(f"  {T:.6f}   {k:2d}  {l:2d}  {l1: .6f}   {fid:.10f}   {im: .1e}")
    T, k, l, l1, fid, im = sols[0]
    print(f"\nglobal minimum: T = {T:.6f}  = {T / (OmB / OMEGA):.3f} x free time (Thm 1: >= 1)")

    qc = qb_circuit(T, l1)
    f = abs(np.vdot(target_state(phi, OmB), Statevector(qc).data)) ** 2
    print(f"Qiskit circuit fidelity with |psi_f>: {f:.10f}\n")
    print(qc.draw(output="text"))


# ----------------------------------------------------------------------------
# 6. General QB for ANY initial and final single-qubit state (restricted operator: sigma_z)
# ----------------------------------------------------------------------------
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize
from qiskit.circuit.library import StatePreparation


def _as_state(psi):
    """Accept a list/array of 2 amplitudes or a qiskit Statevector; return a normalized vector."""
    v = np.asarray(psi.data if isinstance(psi, Statevector) else psi, dtype=complex).ravel()
    if v.shape != (2,):
        raise ValueError("this implementation handles a single qubit (2 amplitudes)")
    return v / np.linalg.norm(v)


def _bloch(psi):
    return np.real([np.vdot(psi, P @ psi) for P in (X, Y, Z)])


def _su2(vec, t):
    """exp(-i t vec.sigma) for a real 3-vector."""
    n = np.linalg.norm(vec)
    if n < 1e-14:
        return I2.copy()
    return np.cos(n * t) * I2 - 1j * np.sin(n * t) * (vec[0] * X + vec[1] * Y + vec[2] * Z) / n


@dataclass
class QBSolution:
    """One extremal of the M=1 closed-subalgebra family (Theorem 4 with lambda_0 = 1):
         H(0) = omega (cos(alpha) X + sin(alpha) Y),     G = lam1 * Z
         H(t) = e^{iGt} H(0) e^{-iGt},   U(t) = e^{iGt} e^{-i[H(0)+G]t}."""
    T: float
    alpha: float
    lam1: float
    omega: float
    psi_i: np.ndarray
    psi_f: np.ndarray
    fidelity: float
    eq55_ratio: float  # Im<HF>/Re<HF> at T; the paper's Eq. (55) asks for 0 (informational only)

    def H0(self):
        return self.omega * (np.cos(self.alpha) * X + np.sin(self.alpha) * Y)

    def U(self, t):
        return expm(1j * self.lam1 * Z * t) @ expm(-1j * (self.H0() + self.lam1 * Z) * t)

    def H(self, t):
        R = expm(1j * self.lam1 * Z * t)
        return R @ self.H0() @ R.conj().T

    def circuit(self):
        """|0> -> |psi_i>, then exp(-i[H(0)+lam1 Z]T), then exp(+i lam1 Z T) = RZ(-2 lam1 T)."""
        qc = QuantumCircuit(1)
        qc.append(StatePreparation(self.psi_i), [0])
        gen = SparsePauliOp.from_list([("X", self.omega * np.cos(self.alpha)),
                                       ("Y", self.omega * np.sin(self.alpha)),
                                       ("Z", self.lam1)])
        qc.append(PauliEvolutionGate(gen, time=self.T), [0])
        qc.rz(-2 * self.lam1 * self.T, 0)
        return qc


def quantum_brachistochrone(psi_i, psi_f, omega=OMEGA, n_starts=400, T_max=None, seed=0, tol=1e-9):
    """Minimum-time evolution from |psi_i> to |psi_f> (up to a phase) for one qubit with
    ||H||: (1/2)Tr H^2 = omega^2, Tr H = 0 and the restricted operator Tr(H sigma_z) = 0,
    i.e. H(t) = h_x X + h_y Y.

    Unknowns (T, alpha, lam1).  Equations:
        - target:  Bloch vector of U(T)|psi_i>  =  Bloch vector of |psi_f>        (2 independent)
        - Eq. (14): <psi_i|F(0)|psi_i> = 0,  F(0) = H(0) + lam1 Z                 (1)
    Both signs of H(0) (alpha and alpha+pi) are covered because alpha is free.
    The eq. (55) residual of each solution is reported but NOT used as a filter (see NOTE above).

    Returns a dict with all distinct solutions found (sorted by T), the shortest one ('best'),
    the free-evolution time Omega_B/omega, and the slowdown relative to free evolution.
    Solutions are local extremals; the shortest found is the candidate global minimum
    (confirm independently with `numerical_min_time`).
    """
    psi_i, psi_f = _as_state(psi_i), _as_state(psi_f)
    s_i, s_f = _bloch(psi_i), _bloch(psi_f)
    OmB = np.arccos(np.clip(abs(np.vdot(psi_i, psi_f)), 0, 1))  # Bures angle
    free_time = OmB / omega
    if OmB < 1e-12:
        return dict(solutions=[], best=None, free_time=0.0, slowdown=1.0, bures_angle=0.0)

    def residual(p):
        T, a, l1 = p
        h = omega * np.array([np.cos(a), np.sin(a), 0.0])
        Hv = h + np.array([0, 0, l1])  # H(0)+G as a Bloch-type vector
        # U = e^{i l1 Z T} e^{-i (h + l1 Z) T}
        U = expm(1j * l1 * Z * T) @ _su2(Hv, T)
        s = _bloch(U @ psi_i)
        return np.r_[s - s_f, h @ s_i + l1 * s_i[2]]  # Bloch mismatch (3), Eq. (14) (1)

    rng = np.random.default_rng(seed)
    T_max = T_max or 2 * np.pi / omega
    found = {}
    for _ in range(n_starts):
        p0 = [rng.uniform(1e-3, T_max), rng.uniform(0, 2 * np.pi), rng.uniform(-6 * omega, 6 * omega)]
        r = least_squares(residual, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        T, a, l1 = r.x
        if r.cost < tol**2 and T > 1e-9:
            key = (round(T, 6), round(l1, 4), round(a % (2 * np.pi), 4))
            found.setdefault(key, (T, a % (2 * np.pi), l1))

    sols = []
    for T, a, l1 in found.values():
        sol = QBSolution(T, a, l1, omega, psi_i, psi_f, 0.0, 0.0)
        psi_T = sol.U(T) @ psi_i
        sol.fidelity = abs(np.vdot(psi_f, psi_T)) ** 2
        HF = sol.H(T) @ (sol.U(T) @ (sol.H0() + l1 * Z) @ sol.U(T).conj().T)
        v = psi_T.conj() @ HF @ psi_T
        sol.eq55_ratio = v.imag / v.real if abs(v.real) > 1e-12 else np.nan
        sols.append(sol)
    sols.sort(key=lambda s: s.T)
    best = sols[0] if sols else None
    return dict(solutions=sols, best=best, free_time=free_time,
                slowdown=(best.T / free_time if best else np.nan), bures_angle=OmB)


def numerical_min_time(psi_i, psi_f, omega=OMEGA, n_seg=24, starts=10, tol=2e-4, seed=1):
    """Independent check: direct optimal control.  H(t) = omega (cos a(t) X + sin a(t) Y) with
    a(t) piecewise constant; bisect on T for the smallest T at which the target is reachable.
    (Upper bound on the true minimum; slow but assumption-free.)"""
    psi_i, psi_f = _as_state(psi_i), _as_state(psi_f)
    rng = np.random.default_rng(seed)

    def evolve(alphas, T):
        psi, dt = psi_i.copy(), T / len(alphas)
        for a in alphas:
            psi = _su2(omega * np.array([np.cos(a), np.sin(a), 0.0]), dt) @ psi
        return psi

    def reachable(T):
        for _ in range(starts):
            r = minimize(lambda a: 1 - abs(np.vdot(psi_f, evolve(a, T))) ** 2,
                         rng.uniform(0, 2 * np.pi, n_seg), method="BFGS",
                         options=dict(gtol=1e-10, maxiter=300))
            if r.fun < 1e-6:
                return True
        return False

    OmB = np.arccos(np.clip(abs(np.vdot(psi_i, psi_f)), 0, 1))
    lo, hi = 0.9 * OmB / omega, 4 * np.pi / omega
    if not reachable(hi):
        return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if reachable(mid):
            hi = mid
        else:
            lo = mid
    return hi


def demo_general():
    print("=== General QB: any initial / final state (omega = %g) ===" % OMEGA)
    ket = lambda t, p: np.array([np.cos(t / 2), np.exp(1j * p) * np.sin(t / 2)])  # Bloch angles
    cases = {
        "|+x> -> (phi=1.1, Omega_B=0.5) [paper's family, Eq.55 NOT satisfiable]":
            (PSI_I, target_state(1.1, 0.5)),
        "|+x> -> on-curve target (Eq.55 satisfiable)":
            (PSI_I, target_state(2.73154, 0.55230)),
        "|0> -> |+y>": (ket(0, 0), ket(np.pi / 2, np.pi / 2)),
        "random -> random": (ket(1.0, 0.4), ket(2.3, 2.0)),
    }
    for name, (pi_, pf_) in cases.items():
        res = quantum_brachistochrone(pi_, pf_)
        b = res["best"]
        print(f"\n{name}")
        if b is None:
            print("   no extremal found (states equal, or none in this family)")
            continue
        print(f"   free time Omega_B/omega = {res['free_time']:.6f};  extremals found: {len(res['solutions'])}")
        print(f"   shortest: T = {b.T:.6f} ({res['slowdown']:.3f} x free)  alpha = {b.alpha:.4f}"
              f"  lambda1 = {b.lam1:.4f}  fidelity = {b.fidelity:.10f}")
        print(f"   Eq.(55) residual Im/Re = {b.eq55_ratio:+.3e}   (0 would satisfy the paper's Eq. 55)")
        f = abs(np.vdot(pf_ / np.linalg.norm(pf_), Statevector(b.circuit()).data)) ** 2
        print(f"   Qiskit circuit fidelity = {f:.10f}")
        Tnum = numerical_min_time(pi_, pf_)
        print(f"   direct optimal-control min time ~ {Tnum:.5f}   (agreement: {abs(Tnum - b.T) < 1e-3})")


# ----------------------------------------------------------------------------
# 6. General solver: ANY initial and final state, M = 1 (restricted operator sigma_z)
# ----------------------------------------------------------------------------
# Unknowns: alpha (direction of H(0) in the xy-plane), lambda1, T, with
#     H(0) = omega (cos(alpha) X + sin(alpha) Y),   F(0) = H(0) + lambda1 Z      (lambda_0 = 1)
#     U(T) = exp(i lambda1 Z T) exp(-i (H(0) + lambda1 Z) T)                    (Eqs. 87-88, generalised)
# Equations: Eq. (14)  <psi_i|F(0)|psi_i> = 0   (for a qubit, {F,P} = F  <=>  this single condition)
#            U(T)|psi_i> ~ |psi_f>               (two real conditions on the Bloch sphere)
# Eq. (55) is NOT imposed; it is evaluated afterwards and reported (see the notes in the docstring).
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize


def bloch_vector(psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.real([np.vdot(psi, P @ psi) for P in (X, Y, Z)])


def _H0(alpha, w=OMEGA):
    return w * (np.cos(alpha) * X + np.sin(alpha) * Y)


def U_general(T, alpha, l1, w=OMEGA):
    """Closed form of exp(i l1 Z T) exp(-i (H0 + l1 Z) T)."""
    A = _H0(alpha, w) + l1 * Z
    Lam = np.sqrt(w**2 + l1**2)  # A^2 = Lam^2 * I
    expA = np.cos(Lam * T) * I2 - 1j * np.sin(Lam * T) * A / Lam
    return (np.cos(l1 * T) * I2 + 1j * np.sin(l1 * T) * Z) @ expA


def _residual(p, n, m, w):
    alpha, l1, T = p
    psi_T = U_general(T, alpha, l1, w) @ _psi_from_bloch_cache["psi"]
    g14 = w * (np.cos(alpha) * n[0] + np.sin(alpha) * n[1]) + l1 * n[2]
    return np.r_[bloch_vector(psi_T) - m, g14]


_psi_from_bloch_cache = {}


@dataclass
class QBSolution:
    T: float
    alpha: float  # H(0) = omega (cos alpha X + sin alpha Y)
    lambda1: float  # Lagrange multiplier of the forbidden operator sigma_z (lambda_0 = 1)
    omega: float
    fidelity: float  # |<psi_f|U(T)|psi_i>|^2
    re_HF: float  # Re<psi_f|H(T)F(T)|psi_f>   (= omega^2 here; sets the normalisation, Eq. 54)
    eq55: float  # Im<psi_f|H(T)F(T)|psi_f> / Re<...>   (0 <=> paper's Eq. 55 satisfied)

    @property
    def satisfies_eq55(self):
        return abs(self.eq55) < 1e-6

    @property
    def mu(self):
        return self.omega * np.cos(self.alpha), self.omega * np.sin(self.alpha)


def qb_solve_m1(psi_i, psi_f, w=OMEGA, ntry=300, Tmax=None, seed=0, tol=1e-9):
    """All extremal trajectories of the M = 1 quantum brachistochrone (X_1 = sigma_z) between
    ANY two pure qubit states, found by multistart root-finding on (alpha, lambda1, T).

    Returns a list of QBSolution sorted by T (the first entry is the fastest candidate found).
    Each entry carries `eq55`, the value of the paper's extra endpoint quantity
    Im<psi_f|H(T)F(T)|psi_f> / Re<psi_f|H(T)F(T)|psi_f>.

    Notes
    -----
    * Only the equations of motion + Eq. (14) + reaching the target are imposed.  In the paper's
      Fig. 2(b) only entries with eq55 == 0 count as locally time-extremal; see
      `optimal_control_check` for an independent test.  For some targets (e.g. |+x> -> phi=1.1,
      Omega_B=0.5) no entry has eq55 == 0, yet the fastest candidate coincides with the minimum
      time found by brute-force optimal control.
    * lambda_0 = 1 is assumed (normal extremals); lambda_0 = 0 branches are not searched.
    * Rz symmetry: rotating both states about z rotates alpha by the same angle.
    """
    psi_i = np.asarray(psi_i, dtype=complex)
    psi_i = psi_i / np.linalg.norm(psi_i)
    psi_f = np.asarray(psi_f, dtype=complex)
    psi_f = psi_f / np.linalg.norm(psi_f)
    if abs(np.vdot(psi_i, psi_f)) > 1 - 1e-12:
        return [QBSolution(0.0, 0.0, 0.0, w, 1.0, w**2, 0.0)]  # already at the target
    Tmax = Tmax if Tmax is not None else 2 * np.pi / w
    n, m = bloch_vector(psi_i), bloch_vector(psi_f)
    _psi_from_bloch_cache["psi"] = psi_i
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(ntry):
        p0 = [rng.uniform(-np.pi, np.pi), rng.uniform(-6 * w, 6 * w), rng.uniform(0.02, 1.0) * Tmax]
        r = least_squares(_residual, p0, args=(n, m, w), xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          bounds=([-10, -40 * w, 1e-6], [10, 40 * w, 1.5 * Tmax]))
        if r.cost < tol**2:
            alpha, l1, T = r.x
            raw.append((T, np.mod(alpha, 2 * np.pi), l1))
    out = []
    for T, alpha, l1 in sorted(raw):
        if any(abs(T - s.T) < 1e-6 and abs(l1 - s.lambda1) < 1e-3
               and abs(np.mod(alpha - s.alpha + np.pi, 2 * np.pi) - np.pi) < 1e-4 for s in out):
            continue
        U = U_general(T, alpha, l1, w)
        dt = 1e-6  # H(T) = i U' U^dag
        Ht = 1j * (U_general(T + dt, alpha, l1, w) - U_general(T - dt, alpha, l1, w)) / (2 * dt) @ U.conj().T
        Ft = U @ (_H0(alpha, w) + l1 * Z) @ U.conj().T
        v = (U @ psi_i).conj() @ Ht @ Ft @ (U @ psi_i)
        fid = abs(np.vdot(psi_f, U @ psi_i)) ** 2
        out.append(QBSolution(T, alpha, l1, w, fid, v.real, v.imag / v.real))
    return out


def qb_circuit_general(sol, psi_i):
    """Qiskit circuit that prepares |psi_i> and applies the QB evolution U(T) of `sol`."""
    psi_i = np.asarray(psi_i, dtype=complex)
    qc = QuantumCircuit(1)
    qc.prepare_state(psi_i / np.linalg.norm(psi_i), 0)
    mux, muy = sol.mu
    gen = SparsePauliOp.from_list([("X", mux), ("Y", muy), ("Z", sol.lambda1)])
    qc.append(PauliEvolutionGate(gen, time=sol.T), [0])  # exp(-i (H(0) + l1 Z) T)
    qc.rz(-2 * sol.lambda1 * sol.T, 0)  # exp(+i l1 Z T)
    return qc


# ---- independent check: brute-force time-optimal control --------------------------------
def _best_fidelity(T, psi_i, psi_f, w=OMEGA, N=30, nstart=12, seed=1):
    """Max over piecewise-constant controls H_j = w(cos th_j X + sin th_j Y) of |<psi_f|psi(T)>|^2.
    (Saturating the norm is WLOG: a smaller norm can always be rescaled to a shorter time.)"""
    rng = np.random.default_rng(seed)
    dt = T / N

    def infid(theta):
        psi = psi_i
        for th in theta:
            Hn = np.cos(th) * X + np.sin(th) * Y
            psi = (np.cos(w * dt) * I2 - 1j * np.sin(w * dt) * Hn) @ psi
        return 1 - abs(np.vdot(psi_f, psi)) ** 2

    best = 0.0
    for _ in range(nstart):
        r = minimize(infid, rng.uniform(-np.pi, np.pi, N), method="BFGS")
        best = max(best, 1 - r.fun)
        if best > 1 - 1e-9:
            break
    return best


def optimal_control_check(psi_i, psi_f, T, w=OMEGA, N=40, nstart=20):
    """Best fidelity reachable in time T, and in time 0.99*T, with norm-bounded controls
    (no QB assumptions).  If T is the true minimum, the first is ~1 and the second is < 1."""
    psi_i = np.asarray(psi_i, dtype=complex) / np.linalg.norm(psi_i)
    psi_f = np.asarray(psi_f, dtype=complex) / np.linalg.norm(psi_f)
    return (_best_fidelity(T, psi_i, psi_f, w, N, nstart),
            _best_fidelity(0.99 * T, psi_i, psi_f, w, N, nstart))


def demo_general():
    print("=== General solver: qb_solve_m1(psi_i, psi_f) ===")
    rng = np.random.default_rng(7)

    def rand_state():
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        return v / np.linalg.norm(v)

    cases = {
        "|+x> -> (phi=1.1, Omega_B=0.5)  [Example A family]": (PSI_I, target_state(1.1, 0.5)),
        "|0>  -> |+x>                    [sigma_z-invariant start]": (np.array([1, 0], complex), PSI_I),
        "random state -> random state": (rand_state(), rand_state()),
    }
    for name, (pi_, pf_) in cases.items():
        print(f"\n--- {name} ---")
        free = np.arccos(min(1.0, abs(np.vdot(pi_, pf_)))) / OMEGA
        sols = qb_solve_m1(pi_, pf_)
        print(f"unrestricted minimum time Omega_B/omega = {free:.6f}   ({len(sols)} candidate(s) found)")
        print("   T          alpha     lambda1     fidelity     Im/Re<HF> (Eq.55)")
        for s in sols[:5]:
            tag = "  <- Eq.55 ok" if s.satisfies_eq55 else ""
            print(f"  {s.T:.6f}  {s.alpha:7.4f}  {s.lambda1:9.4f}  {s.fidelity:.9f}  {s.eq55: .3e}{tag}")
        best = sols[0]
        f_T, f_less = optimal_control_check(pi_, pf_, best.T)
        print(f"fastest candidate T = {best.T:.6f} ({best.T / free:.3f} x free time)")
        print(f"  optimal-control check: best fidelity at T = {f_T:.7f},  at 0.99 T = {f_less:.7f}")
        qc = qb_circuit_general(best, pi_)
        f_c = abs(np.vdot(pf_, Statevector(qc).data)) ** 2
        print(f"  Qiskit circuit fidelity with |psi_f>: {f_c:.9f}")


if __name__ == "__main__":
    run_checks()
    make_figure()
    demo_known_target()
    demo_general()
