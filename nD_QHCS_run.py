# python nD_QHCS_run.py
from qhcs import qhcs_linear_circuit
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit

# ----- physical + simulation parameters -----
n, k   = 4, 1          # 4‑dim example
A, B   = 40.0, 30.0    # in the chaotic window
dt     = 0.04
steps  = 3
shots  = 10_000

# ----- build the evolution circuit (no measurements yet) -----
qc = qhcs_linear_circuit(n, k, A, B, dt, steps, measure=False)

# ----- choose |+…+> initial state (optional) -----
for i in range(n):
    qc.h(qc.qregs[i][0])

# ----- add measurements -----
creg = qc.cregs[0]
for i in range(n):
    qc.measure(qc.qregs[i][0], creg[i])

print(qc.draw("text"))

# ----- run on AerSimulator -----
sim = AerSimulator()
result  = sim.run(qc, shots=shots).result()
counts  = result.get_counts()

# ----- post‑process counts into random numbers -----
rand_list = []
for bitstring, freq in counts.items():
    value = int(bitstring, 2)
    rand_list.extend([value] * freq)

np.random.shuffle(rand_list)
print(f"Generated {len(rand_list)} numbers, first 10: {rand_list[:10]}")

np.savetxt("quantum_random_numbers.txt", rand_list, fmt="%d")
print("Saved to quantum_random_numbers.txt")

circuit_drawer(qc, output='mpl', filename='qhcs_circuit.png')
print("Quantum circuit saved as qhcs_circuit.png")

