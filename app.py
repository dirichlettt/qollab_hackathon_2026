"""Streamlit UI for the single-qubit restricted quantum brachistochrone (Yang & del Campo,
arXiv:2204.12792, Example A).  Run:  streamlit run app.py

Everything is computed locally with Qiskit (statevector + Aer simulator).  The physics lives in
qb_example_A.py; this file only adds controls, tables and plots on top of it.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import streamlit as st
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import qb_example_A as qb

st.set_page_config(page_title="Quantum Brachistochrone Explorer", page_icon="⏱", layout="wide")

st.title("Quantum Brachistochrone Explorer")
st.markdown(
    "Fastest way to steer one qubit when the **σ_z control is forbidden**. "
    "Reproduces Example A of Yang & del Campo, "
    "[arXiv:2204.12792](https://arxiv.org/abs/2204.12792), with Qiskit circuits "
    "verified on a local simulator."
)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    omega = st.slider(
        "Hamiltonian norm bound ω", min_value=1.0, max_value=20.0, value=10.0, step=0.5,
        help="Largest allowed energy scale. The paper's Fig. 2 uses ω = 10.",
    )
    mode = st.radio(
        "How to choose the target state",
        ["From an extremal trajectory (k, λ₁)", "Enter (φ, Ω_B) directly"],
        help="Only targets lying on an extremal curve are reachable by a brachistochrone "
             "of this form. The first option guarantees that.",
    )
    if mode.startswith("From"):
        k0 = st.slider("Branch index k", 0, 4, 0, help="Integer branch in Eq. (103).")
        l1_0 = st.slider("Lagrange multiplier λ₁", -15.0, 15.0, 7.0, 0.5,
                         help="Multiplier enforcing the forbidden σ_z control.")
        T0 = (np.pi / 4 + k0 * np.pi / 2) / np.hypot(l1_0, omega)
        tol = 1e-6
        phi, OmB = qb.target_params(qb.U_t(T0, l1_0, omega) @ qb.PSI_I)
    else:
        phi = st.number_input("Phase φ (radians)", 0.0, float(np.pi), 2.73154, 0.001, format="%.5f")
        OmB = st.number_input("Bures angle Ω_B (radians)", 0.001, float(np.pi / 2), 0.55230, 0.001,
                              format="%.5f",
                              help="Angle between initial and target state. Free minimum time is Ω_B/ω.")
        tol = st.select_slider("Matching tolerance for Eqs. (99)-(101)", [1e-6, 1e-4, 1e-2, 1e-1], 1e-2,
                               help="Looser tolerance admits approximate branches; check the fidelity column.")
    shots = st.select_slider("Aer simulator shots", [100, 1000, 4000, 10000], 1000)

# ----------------------------------------------------------------------------- solve
psi_f = qb.target_state(phi, OmB)
t_free = OmB / omega
sols = qb.solve_known_target(phi, OmB, w=omega, tol=tol)

st.subheader("Target and minimum times")
c1, c2, c3 = st.columns(3)
c1.metric("Target φ", f"{phi:.4f} rad")
c2.metric("Target Ω_B", f"{OmB:.4f} rad")
c3.metric("Unrestricted minimum time Ω_B/ω", f"{t_free:.5f}")

if not sols:
    st.warning(
        "No extremal trajectory of the closed-subalgebra form reaches this target "
        "(Eqs. 99-101 have no integer solution (k, l)). Try the first target mode, "
        "or move the sliders slightly.", icon="⚠️",
    )
    st.stop()

T, k, l, l1, fid, im = sols[0]
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
    hide_index=True, width='stretch',
)

# ----------------------------------------------------------------------------- circuit
st.subheader("Qiskit circuit for U(T)")
qc = qb.qb_circuit(T, l1, omega)
sim = AerSimulator()
tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"])
sv = Statevector(qc)
fid_circ = abs(np.vdot(psi_f, sv.data)) ** 2

left, right = st.columns([3, 2])
with left:
    st.code(qc.draw(output="text"), language="text")
    st.caption(f"Transpiled to rz/sx/x: {tqc.size()} gates, depth {tqc.depth()}. "
               f"Statevector fidelity with target: {fid_circ:.10f}")
with right:
    meas = tqc.copy()
    meas.measure_all()
    counts = sim.run(meas, shots=shots).result().get_counts()
    probs_exact = {"0": abs(psi_f[0]) ** 2, "1": abs(psi_f[1]) ** 2}
    st.markdown(f"**Aer simulation, {shots} shots** (computational basis)")
    st.dataframe(
        [{"outcome": b, "measured fraction": counts.get(b, 0) / shots,
          "exact |amplitude|²": round(probs_exact[b], 4)} for b in ("0", "1")],
        hide_index=True, width='stretch',
    )

# ----------------------------------------------------------------------------- Bloch path
st.subheader("Trajectory on the Bloch sphere")
ts = np.linspace(0, T, 200)
bloch = np.array([
    [np.real(np.conj(p) @ P @ p) for P in (qb.X, qb.Y, qb.Z)]
    for p in (qb.U_t(t, l1, omega) @ qb.PSI_I for t in ts)
])
tf = np.array([np.real(np.conj(psi_f) @ P @ psi_f) for P in (qb.X, qb.Y, qb.Z)])

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
st.pyplot(fig, width='stretch')
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
                 width='stretch')
