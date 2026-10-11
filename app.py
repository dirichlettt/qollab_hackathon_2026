"""Streamlit UI for the single-qubit restricted quantum brachistochrone (Yang & del Campo,
arXiv:2204.12792, Example A, plus the general any-state solver).  Run:  streamlit run app.py

Everything is computed locally with Qiskit (statevector + Aer simulator).  The physics lives in
qb_example_A.py; this file only adds controls, tables and plots on top of it.
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import streamlit as st

import qb_example_A as qb

HERE = os.path.dirname(os.path.abspath(__file__))


def _flat(psi):
    psi = np.asarray(psi, dtype=complex)
    return [float(psi[0].real), float(psi[0].imag), float(psi[1].real), float(psi[1].imag)]


@st.cache_data(show_spinner="Building and simulating the Qiskit circuit...")
def run_qiskit(T, omega, alpha, lambda1, shots, psi_i, psi_f):
    """All Qiskit work happens in a fresh interpreter: see qb_worker.py for why."""
    req = dict(T=T, omega=omega, alpha=alpha, lambda1=lambda1, shots=shots, psi_i=psi_i, psi_f=psi_f)
    proc = subprocess.run([sys.executable, os.path.join(HERE, "qb_worker.py")], input=json.dumps(req),
                          capture_output=True, text=True, cwd=HERE, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout)


@st.cache_data(show_spinner="Searching extremal trajectories (multistart root finding)...")
def solve_general(psi_i, psi_f, omega, ntry):
    sols = qb.qb_solve_m1(np.array(psi_i[::2]) + 1j * np.array(psi_i[1::2]),
                          np.array(psi_f[::2]) + 1j * np.array(psi_f[1::2]), w=omega, ntry=ntry)
    return [dict(T=s.T, alpha=s.alpha, lambda1=s.lambda1, fidelity=s.fidelity, eq55=s.eq55) for s in sols]


def ket(theta, phi):
    """Bloch-sphere angles -> state  cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


st.set_page_config(page_title="Quantum Brachistochrone Explorer", page_icon="⏱", layout="wide")

st.title("Quantum Brachistochrone Explorer")
st.markdown(
    "Fastest way to steer one qubit when the **σ_z control is forbidden**. "
    "Reproduces Example A of Yang & del Campo, "
    "[arXiv:2204.12792](https://arxiv.org/abs/2204.12792), extends it to arbitrary initial and "
    "final states, and verifies every result with a Qiskit circuit on a local simulator."
)

MODE_A, MODE_DIRECT, MODE_GENERAL = (
    "Example A: target from an extremal trajectory (k, λ₁)",
    "Example A: enter target (φ, Ω_B) directly",
    "Any initial and final state (general solver)",
)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    omega = st.slider(
        "Hamiltonian norm bound ω", min_value=1.0, max_value=20.0, value=10.0, step=0.5,
        help="Largest allowed energy scale. The paper's Fig. 2 uses ω = 10.",
    )
    mode = st.radio(
        "Problem to solve", [MODE_A, MODE_DIRECT, MODE_GENERAL],
        help="Example A starts from |+x⟩ and uses the paper's closed forms. The general solver "
             "starts from any state and finds extremals numerically.",
    )
    if mode == MODE_A:
        k0 = st.slider("Branch index k", 0, 4, 0, help="Integer branch in Eq. (103).")
        l1_0 = st.slider("Lagrange multiplier λ₁", -15.0, 15.0, 7.0, 0.5,
                         help="Multiplier enforcing the forbidden σ_z control.")
        T0 = (np.pi / 4 + k0 * np.pi / 2) / np.hypot(l1_0, omega)
        phi, OmB = qb.target_params(qb.U_t(T0, l1_0, omega) @ qb.PSI_I)
        tol = 1e-6
    elif mode == MODE_DIRECT:
        phi = st.number_input("Phase φ (radians)", 0.0, float(np.pi), 2.73154, 0.001, format="%.5f")
        OmB = st.number_input("Bures angle Ω_B (radians)", 0.001, float(np.pi / 2), 0.55230, 0.001,
                              format="%.5f",
                              help="Angle between initial and target state. Free minimum time is Ω_B/ω.")
        tol = st.select_slider("Matching tolerance for Eq. (105)", [1e-6, 1e-4, 1e-2, 1e-1], 1e-2,
                               help="Looser tolerance admits approximate branches; check the fidelity column.")
    else:
        st.markdown("**Initial state** (Bloch angles)")
        th_i = st.slider("θ initial", 0.0, float(np.pi), float(np.pi / 2), 0.01, key="thi",
                         help="Polar angle from |0⟩. θ = π/2, φ = 0 is |+x⟩.")
        ph_i = st.slider("φ initial", 0.0, float(2 * np.pi), 0.0, 0.01, key="phi_i")
        st.markdown("**Final state** (Bloch angles)")
        th_f = st.slider("θ final", 0.0, float(np.pi), 2.3, 0.01, key="thf")
        ph_f = st.slider("φ final", 0.0, float(2 * np.pi), 2.0, 0.01, key="phi_f")
        ntry = st.select_slider("Random starts for the root finder", [50, 150, 300], 150,
                                help="More starts find more extremal branches but take longer.")
    shots = st.select_slider("Aer simulator shots", [100, 1000, 4000, 10000], 1000)

# ----------------------------------------------------------------------------- solve
st.subheader("Target and minimum times")

if mode == MODE_GENERAL:
    psi_i, psi_f = ket(th_i, ph_i), ket(th_f, ph_f)
    OmB = float(np.arccos(np.clip(abs(np.vdot(psi_i, psi_f)), 0, 1)))
    t_free = OmB / omega
    c1, c2, c3 = st.columns(3)
    c1.metric("Bloch vector of initial state", "(%.2f, %.2f, %.2f)" % tuple(qb.bloch_vector(psi_i)))
    c2.metric("Bures angle Ω_B", f"{OmB:.4f} rad")
    c3.metric("Unrestricted minimum time Ω_B/ω", f"{t_free:.5f}")
    if OmB < 1e-6:
        st.info("Initial and final states coincide: nothing to do, minimum time is 0.")
        st.stop()
    rows = solve_general(_flat(psi_i), _flat(psi_f), float(omega), int(ntry))
    if not rows:
        st.warning("The root finder found no extremal trajectory. Increase the number of random "
                   "starts or move the sliders slightly.", icon="⚠️")
        st.stop()
    best = rows[0]
    T, alpha, l1 = best["T"], best["alpha"], best["lambda1"]
    c4, c5, c6 = st.columns(3)
    c4.metric("Restricted minimum time T", f"{T:.5f}", help="Fastest extremal found (candidate global minimum).")
    c5.metric("Slow-down factor T / T_free", f"{T / t_free:.3f}",
              help="Theorem 1 of the paper guarantees this is at least 1.")
    c6.metric("H(0) direction α, multiplier λ₁", f"{alpha:.3f}, {l1:.3f}",
              help="H(0) = ω (cos α σ_x + sin α σ_y). Example A is α = π/2.")
    st.markdown("**All extremal branches found (sorted by time).** The last column is the paper's "
                "extra endpoint quantity, Eq. (55); it is reported, not imposed, because for generic "
                "targets no branch satisfies it even though the fastest one matches brute-force "
                "optimal control.")
    st.dataframe(
        [{"T": round(r["T"], 6), "α": round(r["alpha"], 4), "λ₁": round(r["lambda1"], 4),
          "fidelity": round(r["fidelity"], 10), "Im/Re⟨ψ_f|H F|ψ_f⟩ (Eq. 55)": f"{r['eq55']:.2e}",
          "Eq. 55 satisfied": abs(r["eq55"]) < 1e-6} for r in rows[:12]],
        hide_index=True, width="stretch",
    )
    U_of_t = lambda t: qb.U_general(t, alpha, l1, omega)
else:
    psi_i, psi_f = qb.PSI_I, qb.target_state(phi, OmB)
    t_free = OmB / omega
    sols = qb.solve_known_target(phi, OmB, w=omega, tol=tol)
    c1, c2, c3 = st.columns(3)
    c1.metric("Target φ", f"{phi:.4f} rad")
    c2.metric("Target Ω_B", f"{OmB:.4f} rad")
    c3.metric("Unrestricted minimum time Ω_B/ω", f"{t_free:.5f}")
    if not sols:
        st.warning(
            "No extremal trajectory of the closed-subalgebra form reaches this target "
            "(Eq. 105 has no integer solution (k, l)). Try the first mode, loosen the tolerance, "
            "or use the general solver.", icon="⚠️",
        )
        st.stop()
    T, k, l, l1, fid, im = sols[0]
    alpha = float(np.pi / 2)
    c4, c5, c6 = st.columns(3)
    c4.metric("Restricted minimum time T", f"{T:.5f}", help="Eq. (103), smallest allowed (k, l).")
    c5.metric("Slow-down factor T / T_free", f"{T / t_free:.3f}",
              help="Theorem 1 of the paper guarantees this is at least 1.")
    c6.metric("Multiplier λ₁", f"{l1:.4f}")
    st.markdown("**All candidate branches (sorted by time).** Fidelity is the overlap of the "
                "simulated final state with the target. The last column is the endpoint condition "
                "Eq. (55), which must vanish.")
    st.dataframe(
        [{"T": round(s[0], 6), "k": s[1], "l": s[2], "λ₁": round(s[3], 5),
          "fidelity": round(s[4], 10), "Im⟨ψ_f|H F̃|ψ_f⟩": f"{s[5]:.1e}"} for s in sols],
        hide_index=True, width="stretch",
    )
    U_of_t = lambda t: qb.U_t(t, l1, omega)

# ----------------------------------------------------------------------------- circuit
st.subheader("Qiskit circuit for U(T)")
res = run_qiskit(float(T), float(omega), float(alpha), float(l1), int(shots), _flat(psi_i), _flat(psi_f))
counts, probs_exact = res["counts"], res["exact_probs"]

left, right = st.columns([3, 2])
with left:
    st.code(res["drawing"], language="text")
    st.caption(f"Statevector fidelity with target: {res['fidelity']:.10f}. "
               f"Transpiled to rz/sx/x: {res['n_gates']} gates, depth {res['depth']}, "
               f"fidelity {res['fidelity_transpiled']:.10f}.")
    with st.expander("Transpiled circuit"):
        st.code(res["transpiled_drawing"], language="text")
with right:
    st.markdown(f"**Aer simulation, {shots} shots** (computational basis)")
    st.dataframe(
        [{"outcome": b, "measured fraction": counts.get(b, 0) / shots,
          "exact |amplitude|²": round(probs_exact[b], 4)} for b in ("0", "1")],
        hide_index=True, width="stretch",
    )

# ----------------------------------------------------------------------------- Bloch path
st.subheader("Trajectory on the Bloch sphere")
ts = np.linspace(0, T, 200)
bloch = np.array([qb.bloch_vector(U_of_t(t) @ psi_i) for t in ts])
tf = qb.bloch_vector(psi_f)

fig = Figure(figsize=(7, 3.2))
ax = fig.add_subplot()
for i, (name, ls) in enumerate(zip("xyz", ["-", "--", ":"])):
    ax.plot(ts / T, bloch[:, i], ls, lw=2, label=f"⟨σ_{name}⟩")
    ax.plot([1], [tf[i]], "o", color="k", ms=5)
ax.set_xlabel("t / T")
ax.set_ylabel("Bloch component")
ax.set_ylim(-1.05, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="best", ncol=3)
ax.set_title("Line styles distinguish components; black dots mark the target")
fig.tight_layout()
st.pyplot(fig, width="stretch")
st.caption(
    f"Bloch-vector components along the time-optimal path. Target Bloch vector "
    f"({tf[0]:.3f}, {tf[1]:.3f}, {tf[2]:.3f}) is reached at t = T = {T:.5f}."
)

# ----------------------------------------------------------------------------- Fig. 2
with st.expander("Reproduce Fig. 2 of the paper (ω = 10, as in the paper)"):
    st.markdown("Only parameter pairs on the black contour satisfy the endpoint condition "
                "and are locally time-extremal.")
    if st.button("Compute Fig. 2"):
        with st.spinner("Scanning (ω T/π, λ̃₁) grid..."):
            path = "qb_fig2.png"
            qb.make_figure(path)
        st.image(path, caption="Fig. 2: (a) overlap with initial state, (b) endpoint "
                               "condition; black contour marks locally time-extremal pairs.",
                 width="stretch")
