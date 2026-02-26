from __future__ import annotations

import torch

from torq.QLayer import QLayer
from torq.Templates import get_angle_embedding_sigmas

try:
    from .PennyLaneComparison import PennyLaneComparison
except ImportError as exc:
    raise ImportError(
        "PennyLaneComparison not found. Install with `pip install torq-bench[pennylane]`."
    ) from exc


def _select_circuit(pennylane_backend, ansatz_name: str, data_reupload_every: int):
    if data_reupload_every:
        candidates = {
            "basic_entangling": ("data_re_circuit_basic_entangling",),
            "strongly_entangling": ("data_re_circuit_strongly_entangling",),
            "cross_mesh": ("data_re_circuit_cross_mesh",),
            "cross_mesh_2_rots": ("data_re_circuit_cross_mesh_2_rots",),
            "cross_mesh_cx_rot": ("data_re_circuit_cross_mesh_cx_rot",),
            "no_entanglement_ansatz": (
                "data_re_circuit_no_entanglement_ansatz",
                "data_re_circuit_no_entanglement",
            ),
        }
    else:
        candidates = {
            "basic_entangling": ("circuit_basic_entangling",),
            "strongly_entangling": ("circuit_strongly_entangling",),
            "cross_mesh": ("circuit_cross_mesh",),
            "cross_mesh_2_rots": ("circuit_cross_mesh_2_rots",),
            "cross_mesh_cx_rot": ("circuit_cross_mesh_cx_rot",),
            "no_entanglement_ansatz": (
                "circuit_no_entanglement_ansatz",
                "circuit_no_entanglement",
            ),
        }

    for method_name in candidates.get(ansatz_name, ()):
        if hasattr(pennylane_backend, method_name):
            return getattr(pennylane_backend, method_name)()
    return None


class PennyLaneQLayer(QLayer):
    """QLayer wrapper that runs the PennyLane sanity circuit for timing comparisons."""

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 1,
        ansatz_name: str = "basic_entangling",
        config=None,
        weights=None,
        weights_last_layer_data_re=None,
        q_layer_idx: int = 0,
        param_init_dict=None,
        basis_angle_embedding: str = "X",
        pennylane_dev_name: str | None = None,
    ) -> None:
        super().__init__(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_name=ansatz_name,
            config=config,
            weights=weights,
            weights_last_layer_data_re=weights_last_layer_data_re,
            q_layer_idx=q_layer_idx,
            param_init_dict=param_init_dict,
            basis_angle_embedding=basis_angle_embedding,
        )

        if pennylane_dev_name is None:
            pennylane_dev_name = getattr(self.config, "pennylane_dev_name", "default.qubit")
        pennylane_dev_name = pennylane_dev_name or "default.qubit"

        self._penny = PennyLaneComparison(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            weights=self.params,
            weights_last_layer_data_re=getattr(self, "params_last_layer_reupload", None),
            data_reupload_every=self.data_reupload_every,
            basis_angle_embedding=self.basis_angle_embedding,
            pennylane_dev_name=pennylane_dev_name,
            observable=getattr(self, "observable", None),
            measurement_observables=getattr(self, "measurement_observables", None),
            pauli_measurement_chunk_size=getattr(self.config, "pauli_measurement_chunk_size", 8),
        )
        self._qc = _select_circuit(
            self._penny,
            ansatz_name=self.ansatz_name,
            data_reupload_every=self.data_reupload_every,
        )
        if self._qc is None:
            raise ValueError(
                f"PennyLaneQLayer does not support ansatz_name={self.ansatz_name!r} "
                f"with data_reupload_every={self.data_reupload_every}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._scale_angles(x)
        state = self._qc(x)
        return self._penny.measure_state(state).to(torch.float32)

    def _scale_angles(self, angles: torch.Tensor) -> torch.Tensor:
        _, scaled, _, _ = get_angle_embedding_sigmas(
            angles,
            angle_scaling_method=self.angle_scaling_method,
            angle_scaling=self.angle_scaling,
            basis=self.basis_angle_embedding,
        )
        return scaled
