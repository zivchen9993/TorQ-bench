from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import torch

from torq.QLayer import QLayer
from torq.Templates import get_angle_embedding_sigmas

try:
    from .PennyLaneComparison import PennyLaneComparison
except ImportError as exc:
    raise ImportError(
        "PennyLaneComparison not found. Install with `pip install torq-bench[pennylane]`."
    ) from exc


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
        super_kwargs = dict(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_name=ansatz_name,
            config=config,
            weights=weights,
            weights_last_layer_data_re=weights_last_layer_data_re,
            q_layer_idx=q_layer_idx,
            param_init_dict=param_init_dict,
        )
        if "basis_angle_embedding" in inspect.signature(QLayer.__init__).parameters:
            super_kwargs["basis_angle_embedding"] = basis_angle_embedding
        else:
            config_for_super = copy.copy(config) if config is not None else SimpleNamespace()
            setattr(config_for_super, "basis_angle_embedding", basis_angle_embedding)
            super_kwargs["config"] = config_for_super

        super().__init__(**super_kwargs)

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
            observables=getattr(self.config, "observables", None),
            pauli_measurement_chunk_size=getattr(self.config, "pauli_measurement_chunk_size", 8),
            config=self.config,
        )
        self._measurement_qc = self._penny.build_measurement_circuit(self.ansatz_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._scale_angles(x)
        return self._penny._format_batched_measurement_result(self._measurement_qc(x)).to(
            torch.float32
        )

    def _scale_angles(self, angles: torch.Tensor) -> torch.Tensor:
        _, scaled, _, _ = get_angle_embedding_sigmas(
            angles,
            angle_scaling_method=self.angle_scaling_method,
            angle_scaling=self.angle_scaling,
            basis=self.basis_angle_embedding,
        )
        return scaled
