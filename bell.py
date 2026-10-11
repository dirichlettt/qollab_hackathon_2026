from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)        # put qubit 0 in superposition
qc.cx(0, 1)    # entangle it with qubit 1
qc.measure_all()

print(qc.draw())

sim = AerSimulator()
result = sim.run(transpile(qc, sim), shots=1000).result()
print(result.get_counts())