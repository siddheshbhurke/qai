import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Step 1: Create a 5-qubit quantum circuit
n = 5

qc = QuantumCircuit(n, n)

# Step 2: Initial state |00000>
# Qubits are initialized to |0> by default.

# Step 3: Apply Hadamard and controlled phase gates
for j in range(n):

    # Apply Hadamard gate
    qc.h(j)

    # Apply controlled rotation gates
    for k in range(j + 1, n):

        angle = 3.141592653589793 / (2 ** (k - j))

        qc.cp(angle, k, j)

# Step 4: Swap qubits to reverse their order
for i in range(n // 2):
    qc.swap(i, n - i - 1)

# Step 5: Measure all qubits
qc.measure(range(n), range(n))

print("5-Qubit Quantum Fourier Transform Circuit")
print(qc.draw(output="text"))

# Step 6: Execute circuit using simulator
simulator = AerSimulator()

compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=1024
).result()

counts = result.get_counts()

print("\nQFT Measurement Results:")
print(counts)

# Step 7: Plot measurement histogram
plot_histogram(counts)
plt.title("5-Qubit Quantum Fourier Transform")
plt.show()