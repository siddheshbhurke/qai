import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Step 1: Create a depolarizing noise model
noise_model = NoiseModel()

single_qubit_error = depolarizing_error(0.002, 1)
two_qubit_error = depolarizing_error(0.01, 2)

noise_model.add_all_qubit_quantum_error(
    single_qubit_error, ["x", "sx", "rz"]
)

noise_model.add_all_qubit_quantum_error(
    two_qubit_error, ["cx"]
)

# Initialize noisy simulator
simulator = AerSimulator(noise_model=noise_model)

# Step 2: Define benchmarking sequence lengths
sequence_lengths = [1, 2, 4, 8, 16, 32, 64, 100]

survival_probabilities = []

shots = 1024
num_sequences = 10

# Step 3: Generate random Clifford sequences
for length in sequence_lengths:

    total_survival = 0

    for seed in range(num_sequences):

        # Create random Clifford gates
        sequence_circuit = QuantumCircuit(1)

        rng = np.random.default_rng(seed + length)

        for _ in range(length):
            clifford = random_clifford(
                1,
                seed=int(rng.integers(0, 1000000))
            )

            sequence_circuit.compose(
                clifford.to_circuit(),
                inplace=True
            )

        # Step 4: Calculate inverse Clifford operation
        sequence_clifford = Clifford(sequence_circuit)

        inverse_circuit = sequence_clifford.adjoint().to_circuit()

        # Step 5: Create benchmarking circuit
        qc = QuantumCircuit(1, 1)

        qc.compose(sequence_circuit, inplace=True)
        qc.compose(inverse_circuit, inplace=True)

        # Measure final state
        qc.measure(0, 0)

        # Step 6: Execute circuit
        compiled_circuit = transpile(qc, simulator)

        result = simulator.run(
            compiled_circuit,
            shots=shots
        ).result()

        counts = result.get_counts()

        # Calculate probability of returning to |0>
        survival_probability = counts.get("0", 0) / shots

        total_survival += survival_probability

    average_survival = total_survival / num_sequences

    survival_probabilities.append(average_survival)

    print(
        "Sequence Length:",
        length,
        "Survival Probability:",
        round(average_survival, 4)
    )

# Step 7: Fit exponential decay curve
def decay_function(m, A, p, B):
    return A * (p ** m) + B

parameters, covariance = curve_fit(
    decay_function,
    sequence_lengths,
    survival_probabilities,
    p0=[0.5, 0.99, 0.5],
    bounds=([0, 0, 0], [1, 1, 1]),
    maxfev=10000
)

A, p, B = parameters

# Calculate average single-qubit error per Clifford
error_rate = (1 - p) / 2

print("\nRandomized Benchmarking Results")
print("Decay Parameter:", p)
print("Estimated Error Rate:", error_rate)

# Step 8: Plot fidelity decay curve
x_fit = np.linspace(
    min(sequence_lengths),
    max(sequence_lengths),
    200
)

y_fit = decay_function(x_fit, A, p, B)

plt.figure(figsize=(9, 5))

plt.scatter(
    sequence_lengths,
    survival_probabilities,
    label="Experimental Results"
)

plt.plot(
    x_fit,
    y_fit,
    label="Exponential Decay Fit"
)

plt.xlabel("Random Clifford Sequence Length")
plt.ylabel("Survival Probability")
plt.title("Randomized Benchmarking Protocol")
plt.legend()
plt.grid(True)
plt.show()