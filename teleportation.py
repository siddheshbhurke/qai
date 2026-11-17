import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Create quantum registers
q = QuantumRegister(3, "q")

# Two classical bits for Bell measurement
measurement = ClassicalRegister(2, "measurement")

# One classical bit for Bob's final measurement
output = ClassicalRegister(1, "output")

qc = QuantumCircuit(q, measurement, output)

# Step 1: Prepare the unknown quantum state
theta = 1.2
phi = 0.8

qc.ry(theta, q[0])
qc.rz(phi, q[0])

# Step 2: Create entangled Bell pair between Alice and Bob
qc.h(q[1])
qc.cx(q[1], q[2])

# Step 3: Alice performs Bell-state measurement
qc.cx(q[0], q[1])
qc.h(q[0])

# Step 4: Measure Alice's qubits
qc.measure(q[0], measurement[0])
qc.measure(q[1], measurement[1])

# Step 5: Bob applies conditional correction gates

# Apply X gate if measurement bit 1 is 1
with qc.if_test((measurement[1], 1)):
    qc.x(q[2])

# Apply Z gate if measurement bit 0 is 1
with qc.if_test((measurement[0], 1)):
    qc.z(q[2])

# Step 6: Measure Bob's qubit
qc.measure(q[2], output[0])

print("Quantum Teleportation Circuit")
print(qc.draw(output="text"))

# Execute circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

result = simulator.run(
    compiled_circuit,
    shots=1024
).result()

counts = result.get_counts()

print("\nTeleportation Measurement Results:")
print(counts)

# Plot measurement histogram
plot_histogram(counts)
plt.title("Quantum Teleportation Results")
plt.show()