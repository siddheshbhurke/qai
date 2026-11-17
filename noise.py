import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Create quantum registers
data = QuantumRegister(3, "data")
ancilla = QuantumRegister(2, "ancilla")

# Create classical registers
syndrome = ClassicalRegister(2, "syndrome")
output = ClassicalRegister(3, "output")

qc = QuantumCircuit(data, ancilla, syndrome, output)

# Step 1: Initialize logical qubit
# Encode |0> using a three-qubit repetition code
qc.cx(data[0], data[1])
qc.cx(data[0], data[2])

# Step 2: Introduce bit-flip noise on the second qubit
qc.x(data[1])

# Step 3: Detect errors using parity checks
qc.cx(data[0], ancilla[0])
qc.cx(data[1], ancilla[0])

qc.cx(data[1], ancilla[1])
qc.cx(data[2], ancilla[1])

# Step 4: Measure error syndromes
qc.measure(ancilla[0], syndrome[0])
qc.measure(ancilla[1], syndrome[1])

# Step 5: Correct the detected error
# Syndrome values:
# 01 -> Error on qubit 0
# 11 -> Error on qubit 1
# 10 -> Error on qubit 2

with qc.if_test((syndrome, 1)):
    qc.x(data[0])

with qc.if_test((syndrome, 3)):
    qc.x(data[1])

with qc.if_test((syndrome, 2)):
    qc.x(data[2])

# Step 6: Measure corrected data qubits
qc.measure(data, output)

print("Quantum Error Correction Circuit")
print(qc.draw(output="text"))

# Execute circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=1024
).result()

counts = result.get_counts()

print("\nMeasurement Results:")
for state, frequency in counts.items():
    print(state, ":", frequency)

# Extract corrected output register
corrected_counts = {}

for state, frequency in counts.items():
    corrected_state = state.split()[0]
    corrected_counts[corrected_state] = (
        corrected_counts.get(corrected_state, 0) + frequency
    )

print("\nCorrected Output:")
print(corrected_counts)

# Plot results
plt.figure(figsize=(8, 5))
plt.bar(
    corrected_counts.keys(),
    corrected_counts.values(),
    color=["blue"] * len(corrected_counts)
)
plt.xlabel("Corrected Qubit State")
plt.ylabel("Frequency")
plt.title("Quantum Error Correction Results")
plt.show()