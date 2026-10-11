"""Subprocess worker: builds, transpiles and simulates the brachistochrone circuit with Qiskit.

Run by app.py in a fresh interpreter for every update.  Streamlit reruns its script on a new
thread each time a widget changes, and creating Qiskit circuits across those reruns crashes
Qiskit's Rust allocator (segfault in CircuitData::new on Python 3.14 / Qiskit 2.5).  Keeping all
QuantumCircuit objects in a short-lived process avoids that entirely.

Input (JSON on stdin):
    {"T": .., "omega": .., "alpha": .., "lambda1": .., "shots": .., "psi_i": [re0, im0, re1, im1],
     "psi_f": [re0, im0, re1, im1]}
    H(0) = omega (cos(alpha) X + sin(alpha) Y);  Example A of the paper is alpha = pi/2.
Output: JSON on stdout.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm
from qiskit_aer import AerSimulator

PLUS_X = np.array([1, 1]) / np.sqrt(2)


def _vec(v):
    a = np.array([v[0] + 1j * v[1], v[2] + 1j * v[3]])
    return a / np.linalg.norm(a)


def build_circuit(T, omega, alpha, l1, psi_i):
    """|0> -> |psi_i>, then exp(-i (H(0) + l1 Z) T), then exp(+i l1 Z T) = RZ(-2 l1 T).
    The evolution step is appended as the exact 2x2 unitary: X/Y and Z do not commute, so the
    default single-step Trotter split of PauliEvolutionGate would not be exact (fidelity ~0.96),
    and Qiskit 2.5's transpiler rejects PauliEvolutionGate(synthesis=MatrixExponential())."""
    qc = QuantumCircuit(1)
    if abs(abs(np.vdot(PLUS_X, psi_i)) - 1) < 1e-9:
        qc.h(0)  # Example A: |+x>
    else:
        qc.prepare_state(psi_i, 0)
    gen = SparsePauliOp.from_list([("X", omega * np.cos(alpha)), ("Y", omega * np.sin(alpha)), ("Z", l1)])
    qc.append(UnitaryGate(expm(-1j * T * gen.to_matrix()), label="exp(-i(H(0)+λ₁Z)T)"), [0])
    qc.rz(-2 * l1 * T, 0)
    return qc


def main(req):
    psi_i, psi_f = _vec(req["psi_i"]), _vec(req["psi_f"])
    qc = build_circuit(req["T"], req["omega"], req["alpha"], req["lambda1"], psi_i)
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"])
    sv = Statevector(qc).data
    sv_t = Statevector(tqc).data
    meas = tqc.copy()
    meas.measure_all()
    counts = AerSimulator().run(meas, shots=int(req["shots"])).result().get_counts()
    return {
        "drawing": str(qc.draw(output="text")),
        "transpiled_drawing": str(tqc.draw(output="text")),
        "n_gates": tqc.size(),
        "depth": tqc.depth(),
        "fidelity": float(abs(np.vdot(psi_f, sv)) ** 2),
        "fidelity_transpiled": float(abs(np.vdot(psi_f, sv_t)) ** 2),
        "counts": {k: int(v) for k, v in counts.items()},
        "exact_probs": {"0": float(abs(psi_f[0]) ** 2), "1": float(abs(psi_f[1]) ** 2)},
    }


if __name__ == "__main__":
    print(json.dumps(main(json.load(sys.stdin))))
