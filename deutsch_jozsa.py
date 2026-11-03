
from pathlib import Path
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Experiment settings
N = 3
SHOTS = 4096
OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)


def create_oracle(n, function_type, value=0, mask=None):
    oracle = QuantumCircuit(n + 1, name="Uf")

    if function_type == "constant":
        if value == 1:
            oracle.x(n)

    elif function_type == "balanced":
        if mask is None or len(mask) != n:
            raise ValueError("Invalid mask")

        if not any(bit == "1" for bit in mask):
            raise ValueError("Balanced mask cannot be zero")

        for i, bit in enumerate(reversed(mask)):
            if bit == "1":
                oracle.cx(i, n)
    else:
        raise ValueError("Invalid function type")

    return oracle


def create_dj_circuit(n, oracle):
    circuit = QuantumCircuit(n + 1, n)

    # Prepare the auxiliary qubit in |1>
    circuit.x(n)

    # Create superposition
    circuit.h(range(n + 1))
    circuit.barrier()

    # Apply the oracle
    circuit.compose(oracle, inplace=True)
    circuit.barrier()

    # Apply Hadamard to input qubits
    circuit.h(range(n))

    # Measure input qubits
    circuit.measure(range(n), range(n))

    return circuit


experiments = [
    ("constant_0", "constant", 0, None),
    ("constant_1", "constant", 1, None),
    ("balanced_001", "balanced", 0, "001"),
    ("balanced_010", "balanced", 0, "010"),
    ("balanced_011", "balanced", 0, "011"),
    ("balanced_100", "balanced", 0, "100"),
    ("balanced_101", "balanced", 0, "101"),
    ("balanced_111", "balanced", 0, "111"),
]

simulator = AerSimulator()

for name, function_type, value, mask in experiments:
    oracle = create_oracle(N, function_type, value, mask)
    circuit = create_dj_circuit(N, oracle)

    compiled = transpile(circuit, simulator)
    result = simulator.run(
        compiled, shots=SHOTS, seed_simulator=42
    ).result()

    counts = result.get_counts()
    measured = max(counts, key=counts.get)

    classification = (
        "CONSTANT" if measured == "0" * N
        else "BALANCED"
    )

    expected = function_type.upper()
    assert classification == expected, f"Test failed: {name}"

    print(f"{name}: {counts} -> {classification}")

    fig = circuit.draw(output="mpl")
    fig.savefig(OUTPUT / f"{name}_circuit.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plot_histogram(counts, title=name)
    fig.savefig(OUTPUT / f"{name}_histogram.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

print("\nAll 8 experiments completed successfully!")
