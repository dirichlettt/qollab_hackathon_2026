"""Subprocess worker: builds, transpiles and simulates the brachistochrone circuit with Qiskit.

Run by app.py in a fresh interpreter for every update.  Streamlit reruns its script on a new
thread each time a widget changes, and creating Qiskit circuits across those reruns crashes
Qiskit's Rust allocator (segfault in CircuitData::new on Python 3.14 / Qiskit 2.5).  Keeping all
QuantumCircuit objects in a short-lived process avoids that entirely.

Usage:  python qb_worker.py T l1 omega shots phi OmB   ->  JSON on stdout
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import qb_example_A as qb


def main(T, l1, omega, shots, phi, OmB):
    qc = qb.qb_circuit(T, l1, omega)
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"])
    psi_f = qb.target_state(phi, OmB)
    sv = Statevector(qc).data
    meas = tqc.copy()
    meas.measure_all()
    counts = AerSimulator().run(meas, shots=shots).result().get_counts()
    return {
        "drawing": str(qc.draw(output="text")),
        "transpiled_drawing": str(tqc.draw(output="text")),
        "n_gates": tqc.size(),
        "depth": tqc.depth(),
        "fidelity": float(abs(np.vdot(psi_f, sv)) ** 2),
        "counts": {k: int(v) for k, v in counts.items()},
        "exact_probs": {"0": float(abs(psi_f[0]) ** 2), "1": float(abs(psi_f[1]) ** 2)},
    }


if __name__ == "__main__":
    T, l1, omega, shots, phi, OmB = (float(x) for x in sys.argv[1:7])
    print(json.dumps(main(T, l1, omega, int(shots), phi, OmB)))
