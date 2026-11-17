import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Initialize 16 qubits and 16 classical bits
n = 16
qc = QuantumCircuit(n, n)

# Apply Hadamard gate to all qubits
for i in range(n):
    qc.h(i)

# Measure all qubits
qc.measure(range(n), range(n))

# Display circuit
print("16-Qubit Random Number Generator")
print(qc.draw(output="text"))

# Execute circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=1024
).result()

counts = result.get_counts()

# Generate a random 16-bit number from measurement results
random_bits = max(counts, key=counts.get)
random_number = int(random_bits.replace(" ", ""), 2)

print("\nMeasurement Results:")
print("Random Binary Number:", random_bits)
print("Random Decimal Number:", random_number)

# Display measurement histogram
top_results = dict(
    sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
)

plt.figure(figsize=(12, 5))
plt.bar(top_results.keys(), top_results.values())
plt.xlabel("16-bit Random Numbers")
plt.ylabel("Frequency")
plt.title("16-Qubit Quantum Random Number Generator")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()