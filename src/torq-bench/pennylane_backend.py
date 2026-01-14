from __future__ import annotations

import torch

from torq.QLayer import QLayer
from torq.Templates import get_angle_embedding_sigmas

try:
    from utility import PennyLaneSanityCheck as qml
except ImportError as exc:
    raise ImportError(
        "PennyLaneSanityCheck not found. This wrapper expects your original utility "
        "module to be importable in the current environment."
    ) from exc


class PennyLaneQLayer(QLayer):
    """QLayer wrapper that runs the PennyLane sanity circuit for timing comparisons."""

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 1,
        ansatz_name: str = "strongly_entangling",
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

        if getattr(self.config, "data_reupload_every", 0):
            raise ValueError("PennyLaneQLayer does not support data_reupload_every yet.")
        if self.ansatz_name != "strongly_entangling":
            raise ValueError("PennyLaneQLayer only supports ansatz_name='strongly_entangling'.")

        if pennylane_dev_name is None:
            pennylane_dev_name = getattr(self.config, "pennylane_dev_name", None)

        self._penny = qml.qml_sanity_check(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            weights=self.params,
            pennylane_dev_name=pennylane_dev_name,
        )
        self._qc = self._penny.circuit_strongly_entangling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._scale_angles(x)
        return torch.stack(self._qc(x), dim=1).to(torch.float32)

    def _scale_angles(self, angles: torch.Tensor) -> torch.Tensor:
        _, scaled, _, _ = get_angle_embedding_sigmas(
            angles,
            angle_scaling_method=self.angle_scaling_method,
            angle_scaling=self.angle_scaling,
            basis=self.basis_angle_embedding,
        )
        return scaled
