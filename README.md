# TorQ-bench

Benchmarks and PennyLane comparisons for TorQ.

TorQ-bench is a small companion package that lets you run the same layer logic in
TorQ and PennyLane to compare outputs and timing. It intentionally keeps PennyLane
out of the core TorQ package.

## Install

From source (recommended while developing):

```bash
pip install -e .[pennylane]
```

Or with PyPI:

```bash
pip install torq-bench[pennylane]
```

TorQ-bench depends on `torq-quantum>=0.1.2`. PennyLane is optional and only required for
the comparison wrappers.

Note: the PyPI distribution is `torq-quantum`, while the Python import package remains `torq`.

## Quickstart: compare TorQ vs PennyLane

```python
import torch
from torq.QLayer import QLayer
from torq_bench import PennyLaneQLayer

n_qubits = 4
n_layers = 2
x = torch.rand(8, n_qubits)

torq_layer = QLayer(n_qubits=n_qubits, n_layers=n_layers)
pl_layer = PennyLaneQLayer(
    n_qubits=n_qubits,
    n_layers=n_layers,
    pennylane_dev_name="default.qubit",
)

y_torq = torq_layer(x)
y_pl = pl_layer(x)
```

Notes:
- `PennyLaneQLayer` supports TorQ ansatz names:
  `basic_entangling`, `strongly_entangling`, `cross_mesh`, `cross_mesh_2_rots`,
  `cross_mesh_cx_rot`, and `no_entanglement_ansatz`.
- `data_reupload_every` is supported.
- Output observables support `local_observable_name` (including `X`, `Y`, `custom`)
  and `measurement_observables` specs (including non-local terms).

## Using PennyLaneComparison directly

```python
import torch
from torq_bench import PennyLaneComparison

n_qubits = 4
n_layers = 2
weights = torch.rand(n_layers, n_qubits, 3)
x = torch.rand(n_qubits)

qc = PennyLaneComparison(n_qubits=n_qubits, n_layers=n_layers, weights=weights)
circuit = qc.circuit_strongly_entangling()
y = circuit(x)
```

## Run the built-in demo

The comparison module includes a demo that builds and draws several circuits.
It uses `qml.draw_mpl`, so you will need `matplotlib` installed.

```bash
python -m torq_bench.PennyLaneComparison
```

## License

MIT
