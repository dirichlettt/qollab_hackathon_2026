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
  5. for a prescribed target |psi_f(phi, Omega_B)> evaluates the closed forms (98)-(105): the
     compatibility condition (105) picks the allowed (k, l), Eq. (103) gives T(k, l), and the
     result is verified with a Qiskit circuit.  NOTE: (phi, Omega_B) must lie on a curve
     satisfying Eq. (105); a generic target is NOT reached by an extremal trajectory of this form
     (3 conditions - 2 for the state, 1 for Eq. 55 - on 2 unknowns lambda1, T).

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
    axs[1].set_title(r"(b)  Im$\langle\psi_f|H(T)\tilde F(T)|\psi_f\rangle$  (black: Eq. 55 satisfied)")
    for ax in axs:
        ax.set_xlabel(r"$\omega T/\pi$")
    axs[0].set_ylabel(r"$\tilde\lambda_1$")
    fig.colorbar(p0, ax=axs[0])
    fig.colorbar(p1, ax=axs[1])
    fig.suptitle(rf"Example A, $\omega={OMEGA:g}$: only pairs on the black contours are locally time-extremal")
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


if __name__ == "__main__":
    run_checks()
    make_figure()
    demo_known_target()
