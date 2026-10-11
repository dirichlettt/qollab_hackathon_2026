"""
Example A of Yang & del Campo, "Minimum-Time Quantum Control and the Quantum
Brachistochrone Equation" (arXiv:2204.12792): one qubit, restricted operator X1 = sigma_z.

Closed-subalgebra solution (Eqs. 87-88 specialised to X1 = sigma_z, i.e. Eqs. 92-93,
with lambda_0 = 1, mu_x = 0, mu_y = omega):
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
     compatibility conditions (99)-(101) pick the allowed (k, l), Eq. (103) gives T(k, l), and the
     result is verified with a Qiskit circuit.  NOTE: (phi, Omega_B) must lie on a curve
     satisfying Eqs. (99)-(101); a generic target is NOT reached by an extremal trajectory of this
     form (3 conditions - 2 for the state, 1 for Eq. 55 - on 2 unknowns lambda1, T).
     Eq. (105) of the paper is the squared/divided form of (99)-(101); it fixes |sin theta_1| but
     drops a sign that depends on (k, l), so it is checked in absolute value only (see
     solve_known_target).

The propagator exp(-i (omega Y + l1 Z) T) is a rotation about the axis (0, omega, l1)/Lambda1 in
the YZ plane, so it is built exactly from standard gates as Rx(-theta1) Ry(2 Lambda1 T) Rx(theta1)
(Eq. 95: tan theta1 = l1/omega).  A PauliEvolutionGate with its default single-step Lie-Trotter
synthesis is NOT exact here (Y and Z do not commute; fidelity ~0.96 after transpilation for the
demo parameters), and MatrixExponential synthesis trips a bug in the qiskit 2.5 HLS plugin.

Requires: qiskit, numpy, scipy, matplotlib.   Run:  python qb_example_A.py
"""
import itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.quantum_info import Statevector

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
    With Lambda1 = sqrt(w^2 + l1^2) and tan(theta1) = l1/w (Eq. 95),
        wY + l1 Z = Lambda1 (cos theta1 Y + sin theta1 Z) = Lambda1 Rx(theta1) Y Rx(theta1)^dag,
    so  exp(-i (wY + l1 Z) T) = Rx(theta1) Ry(2 Lambda1 T) Rx(-theta1)   (exact, no Trotter error).
    """
    Lam = np.hypot(w, l1)
    th = np.arctan2(l1, w)
    qc = QuantumCircuit(1)
    qc.h(0)  # |0> -> |+x>
    qc.rx(-th, 0)
    qc.ry(2 * Lam * T, 0)
    qc.rx(th, 0)
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
    # the same circuit after transpilation to a hardware basis must still be exact
    tqc = transpile(qb_circuit(T, l1), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    fid_t = abs(np.vdot(psi, Statevector(tqc).data)) ** 2
    print(f"transpiled (rz,sx,x,cx) fidelity       : {fid_t:.12f}   ({tqc.size()} gates)")

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
def endpoint_quantities_vec(T, l1, w=OMEGA):
    """Vectorised endpoint_quantities over broadcastable arrays T, l1, using the closed-form
    2x2 exponentials  exp(i a Z) = cos a + i sin a Z,
                      exp(-i (wY + l1 Z) T) = cos(L T) - i sin(L T) (wY + l1 Z)/L,  L = sqrt(w^2+l1^2).
    Returns (|<psi_f|psi_i>|, c, Im<psi_f|H F~|psi_f>) with the same conventions as above."""
    T = np.asarray(T, float)[..., None, None]
    l1 = np.asarray(l1, float)[..., None, None]
    L = np.hypot(w, l1)
    A = w * Y + l1 * Z
    R = np.cos(l1 * T) * I2 + 1j * np.sin(l1 * T) * Z  # exp(i l1 Z T)
    E = np.cos(L * T) * I2 - 1j * np.sin(L * T) * A / L  # exp(-i A T)
    U = R @ E
    Rd = np.swapaxes(R.conj(), -1, -2)
    Ud = np.swapaxes(U.conj(), -1, -2)
    H = w * R @ Y @ Rd
    F = U @ A @ Ud
    psi_f = U @ PSI_I[:, None]
    val = (np.swapaxes(psi_f.conj(), -1, -2) @ H @ F @ psi_f)[..., 0, 0]
    c = val.real
    amp = np.abs((np.swapaxes(psi_f.conj(), -1, -2) @ PSI_I[:, None])[..., 0, 0])
    return amp, c, (val / c).imag


def make_figure(path="qb_fig2.png", n_l=161, n_T=161):
    # axes of the paper: omega*T/pi in (0.2, 1.0],  lambda~1 = lambda1/c in (0, 0.2]
    wT = np.linspace(0.2, 1.0, n_T)
    lt = np.linspace(0.001, 0.2, n_l)
    Ts = wT * np.pi / OMEGA
    # lambda~1 = lambda1 / c with c = Re<psi_f|H F|psi_f>.  For this example c = omega^2 exactly
    # (Fig. 2 caption of the paper); this is asserted below rather than assumed.
    TT, LL = np.meshgrid(Ts, lt)
    amp, c, imag = endpoint_quantities_vec(TT, LL * OMEGA**2)
    assert np.allclose(c, OMEGA**2), "c = Re<psi_f|H F|psi_f> should equal omega^2 here"
    # spot-check the vectorised closed form against the expm-based functions
    a0, c0, i0 = endpoint_quantities(Ts[37], lt[91] * OMEGA**2)
    assert np.allclose([a0, c0, i0], [amp[91, 37], c[91, 37], imag[91, 37]])

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
    """Closed-form candidates T(k, l) of Eq. (103) that satisfy the compatibility conditions
    Eqs. (99)-(101).  Each is then confirmed against the real dynamics (fidelity and Eq. 55).

    Eq. (105) of the paper is NOT used as the filter: it is (99)-(101) squared and divided, so it
    only constrains |sin theta_1| and drops a (k, l)-dependent sign.  Used literally it rejects
    genuine extremal trajectories (e.g. the one generated by k=1, lambda1=3 at omega=10).  It is
    kept here as an absolute-value sanity check only."""
    psi_f = target_state(phi, OmB)
    ac = arccot(np.sin(phi) * np.tan(2 * OmB))
    s2, c2 = np.sin(2 * OmB), np.cos(2 * OmB)
    lhs_fac = np.sqrt(1 + (1 / np.tan(2 * OmB)) ** 2 / np.sin(phi) ** 2)
    rhs105 = s2 * np.sin(phi)
    out = []
    for k, l in itertools.product(range(0, kmax + 1), range(-lmax, lmax + 1)):
        disc = (np.pi / 4 + k * np.pi / 2) ** 2 - 0.25 * (ac + l * np.pi) ** 2
        if disc <= 0:
            continue
        T = np.sqrt(disc) / abs(w)  # Eq. (103)
        l1 = 0.5 * (ac + l * np.pi) / T  # Eq. (102)
        sgn = (-1) ** k
        sin_th, cos_th = 2 * (ac / np.pi + l) / (1 + 2 * k), w / np.hypot(l1, w)  # Eq. (104), (95)
        res = (sgn * cos_th + np.cos(phi) * s2,  # Eq. (99)
               sgn * sin_th * np.sin(2 * l1 * T) - c2,  # Eq. (100)
               sgn * sin_th * np.cos(2 * l1 * T) - np.sin(phi) * s2)  # Eq. (101)
        if max(abs(r) for r in res) > tol:
            continue
        lhs105 = 2 * (ac / np.pi + l) / ((1 + 2 * k) * lhs_fac)
        assert abs(abs(lhs105) - abs(rhs105)) < tol, "Eq. (105) should hold up to sign"
        fid = abs(np.vdot(psi_f, U_t(T, l1, w) @ PSI_I)) ** 2
        _, _, im = endpoint_quantities(T, l1, w)
        out.append((T, k, l, l1, fid, im))
    return sorted(out)


def roundtrip_test(cases=((0, 7.0), (0, -4.0), (1, 3.0), (1, -12.0), (2, -2.0), (3, 5.0))):
    """Generate a target from a known extremal trajectory (k0, lambda1) and check that
    solve_known_target recovers it.  (With the literal Eq. (105) filter the odd-sign cases,
    e.g. (1, 3.0) and (3, 5.0), were lost.)"""
    print("round-trip of Eqs. (99)-(104):")
    ok = True
    for k0, l1_0 in cases:
        T0 = (np.pi / 4 + k0 * np.pi / 2) / np.hypot(l1_0, OMEGA)
        phi, OmB = target_params(U_t(T0, l1_0) @ PSI_I)
        sols = solve_known_target(phi, OmB)
        hit = any(abs(T - T0) < 1e-6 and abs(l1 - l1_0) < 1e-6 and fid > 1 - 1e-9
                  for T, k, l, l1, fid, im in sols)
        ok &= hit
        print(f"  k={k0} lambda1={l1_0:+6.1f}: recovered={hit}  (candidates: {len(sols)})")
    print("  all recovered:", ok, "\n")


def demo_known_target():
    print("=== Prescribed target (Eqs. 97-105) ===")
    roundtrip_test()
    # A target that lies on an extremal curve: take the k=0, lambda1=7 trajectory.
    k0, l1_0 = 0, 7.0
    T0 = (np.pi / 4 + k0 * np.pi / 2) / np.hypot(l1_0, OMEGA)
    phi, OmB = target_params(U_t(T0, l1_0) @ PSI_I)
    print(f"target from (k=0, lambda1={l1_0}): phi = {phi:.5f}, Omega_B = {OmB:.5f}")
    print(f"unrestricted minimum time Omega_B/omega = {OmB / OMEGA:.6f}")

    sols = solve_known_target(phi, OmB)
    if not sols:
        print("no (k, l) satisfies Eqs. (99)-(101) for this target")
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
