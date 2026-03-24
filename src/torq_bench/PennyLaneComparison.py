from __future__ import annotations

try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError("Install with `pip install torq-bench[pennylane]`") from exc

import sys
from types import SimpleNamespace

import torch
import torq as tq

_ANGLE_EMBEDDING_ROTATIONS = {
    "x": "X",
    "rx": "X",
    "y": "Y",
    "ry": "Y",
    "z": "Z",
    "rz": "Z",
}

_SINGLE_QUBIT_ROTATIONS = {
    "x": qml.RX,
    "rx": qml.RX,
    "y": qml.RY,
    "ry": qml.RY,
    "z": qml.RZ,
    "rz": qml.RZ,
}

_NO_DATA_CIRCUIT_METHODS = {
    "basic_entangling": "circuit_basic_entangling",
    "single_rot_basic_ent": "circuit_single_rot_basic_ent",
    "strongly_entangling": "circuit_strongly_entangling",
    "cross_mesh": "circuit_cross_mesh",
    "cross_mesh_2_rots": "circuit_cross_mesh_2_rots",
    "cross_mesh_cx_rot": "circuit_cross_mesh_cx_rot",
    "tile": "circuit_tile",
    "no_entanglement_ansatz": "circuit_no_entanglement_ansatz",
}

_DATA_RE_CIRCUIT_METHODS = {
    "basic_entangling": "data_re_circuit_basic_entangling",
    "single_rot_basic_ent": "data_re_circuit_single_rot_basic_ent",
    "strongly_entangling": "data_re_circuit_strongly_entangling",
    "cross_mesh": "data_re_circuit_cross_mesh",
    "cross_mesh_2_rots": "data_re_circuit_cross_mesh_2_rots",
    "cross_mesh_cx_rot": "data_re_circuit_cross_mesh_cx_rot",
    "tile": "data_re_circuit_tile",
    "no_entanglement_ansatz": "data_re_circuit_no_entanglement_ansatz",
}


def _resolve_single_rotation_gate(name: str):
    normalized = (name or "rx").lower()
    try:
        return _SINGLE_QUBIT_ROTATIONS[normalized], normalized
    except KeyError as exc:
        raise ValueError(
            "single_rotation_gate must be one of ('x', 'rx', 'y', 'ry', 'z', 'rz'). "
            f"Got: {name!r}"
        ) from exc


class PennyLaneComparison:
    def __init__(
        self,
        n_qubits=3,
        n_layers=1,
        weights=None,
        weights_last_layer_data_re=None,
        data_reupload_every=0,
        pennylane_dev_name="default.qubit",
        basis_angle_embedding="X",
        observables=None,
        pauli_measurement_chunk_size=8,
        config=None,
    ):
        self.device = qml.device(pennylane_dev_name, wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = weights
        self.params_last_layer_reupload = weights_last_layer_data_re
        self.data_reupload_every = data_reupload_every
        self.config = config if config is not None else SimpleNamespace()

        basis_key = (basis_angle_embedding or "X").lower()
        if basis_key not in _ANGLE_EMBEDDING_ROTATIONS:
            raise ValueError(
                "basis_angle_embedding must be one of X/RX, Y/RY, Z/RZ for PennyLaneComparison."
            )
        self._angle_embedding_rotation = _ANGLE_EMBEDDING_ROTATIONS[basis_key]

        self._single_rotation, self._single_rotation_name = _resolve_single_rotation_gate(
            getattr(self.config, "single_rotation_gate", "rx")
        )
        self.tile_rotation_params = getattr(self.config, "tile_rotation_params", 3)
        if self.tile_rotation_params not in (1, 3):
            raise ValueError(
                "tile_rotation_params must be one of (1, 3). "
                f"Got: {self.tile_rotation_params!r}"
            )
        self.tile_sublayers = getattr(self.config, "tile_sublayers", 1)
        if self.tile_sublayers < 1:
            raise ValueError("tile_sublayers must be >= 1.")
        self.tile_cyclic = bool(getattr(self.config, "tile_cyclic", False))

        if pauli_measurement_chunk_size < 1:
            raise ValueError("pauli_measurement_chunk_size must be >= 1.")
        self.pauli_measurement_chunk_size = pauli_measurement_chunk_size
        self.observables = observables

    def _angle_embed(self, x):
        qml.AngleEmbedding(
            x,
            wires=range(self.n_qubits),
            rotation=self._angle_embedding_rotation,
        )

    def _params_no_data_reupload(self):
        params = self.params
        if params is not None and params.ndim >= 2 and params.shape[1] == 1:
            params = params[:, 0]
        return params

    def _main_layer_weights(self, layer: int):
        params = self._params_no_data_reupload()
        if params is None:
            raise ValueError("weights must be provided.")
        return params[layer]

    def _data_re_layer_weights(self, layer: int, rep: int):
        if self.params is None:
            raise ValueError("weights must be provided.")
        return self.params[layer, rep]

    def _last_data_re_weights(self, rep: int):
        if self.params_last_layer_reupload is None:
            raise ValueError(
                "weights_last_layer_data_re must be provided when data_reupload_every > 0."
            )
        return self.params_last_layer_reupload[rep]

    def _cnot_ladder_pairs(self, layer_idx: int):
        if self.n_qubits < 2:
            return ()
        offset = layer_idx % self.n_qubits
        if (offset + 1) == self.n_qubits:
            return ()
        distance = offset + 1
        return tuple(
            (control, (control + distance) % self.n_qubits)
            for control in range(self.n_qubits)
        )

    def _cross_mesh_pairs(self):
        return tuple(
            (control, target)
            for control in reversed(range(self.n_qubits))
            for target in reversed(range(self.n_qubits))
            if control != target
        )

    def _tile_pairs(self):
        pairs = []
        for control in range(0, self.n_qubits - 1, 2):
            pairs.append((control, control + 1))
        for control in range(1, self.n_qubits - 1, 2):
            pairs.append((control, control + 1))
        if self.tile_cyclic and self.n_qubits > 2:
            pairs.append((self.n_qubits - 1, 0))
        return tuple(pairs)

    def _apply_cnot_pairs(self, pairs):
        for control, target in pairs:
            qml.CNOT(wires=[control, target])

    def _apply_rot_wall(self, weights):
        for qubit in range(self.n_qubits):
            qml.Rot(weights[qubit, 0], weights[qubit, 1], weights[qubit, 2], wires=qubit)

    def _apply_single_rotation_wall(self, weights, rotation_gate=None):
        gate = rotation_gate if rotation_gate is not None else self._single_rotation
        for qubit in range(self.n_qubits):
            gate(weights[qubit], wires=qubit)

    def _apply_basic_entangling_layer(self, layer_idx: int, weights):
        del layer_idx
        self._apply_rot_wall(weights)
        self._apply_cnot_pairs(self._cnot_ladder_pairs(0))

    def _apply_single_rot_basic_ent_layer(self, layer_idx: int, weights):
        del layer_idx
        self._apply_single_rotation_wall(weights)
        self._apply_cnot_pairs(self._cnot_ladder_pairs(0))

    def _apply_strongly_entangling_layer(self, layer_idx: int, weights):
        self._apply_rot_wall(weights)
        self._apply_cnot_pairs(self._cnot_ladder_pairs(layer_idx))

    def _apply_cross_mesh_layer(self, layer_idx: int, weights):
        del layer_idx
        pairs = self._cross_mesh_pairs()
        flat = weights.flatten()
        for qubit in range(self.n_qubits):
            qml.RX(flat[qubit], wires=qubit)
        for idx, (control, target) in enumerate(pairs):
            qml.CRZ(flat[self.n_qubits + idx], wires=[control, target])

    def _apply_cross_mesh_2_rots_layer(self, layer_idx: int, weights):
        del layer_idx
        pairs = self._cross_mesh_pairs()
        flat = weights.flatten()
        theta_first = flat[: self.n_qubits]
        theta_second = flat[self.n_qubits : 2 * self.n_qubits]
        for qubit in range(self.n_qubits):
            qml.RX(theta_first[qubit], wires=qubit)
            qml.RZ(theta_second[qubit], wires=qubit)
        for idx, (control, target) in enumerate(pairs):
            qml.CRZ(flat[2 * self.n_qubits + idx], wires=[control, target])

    def _apply_cross_mesh_cx_rot_layer(self, layer_idx: int, weights):
        del layer_idx
        self._apply_rot_wall(weights)
        self._apply_cnot_pairs(self._cross_mesh_pairs())

    def _apply_tile_layer(self, layer_idx: int, weights):
        del layer_idx
        if self.tile_rotation_params == 3:
            self._apply_rot_wall(weights)
        else:
            self._apply_single_rotation_wall(weights)
        tile_pairs = self._tile_pairs()
        for _ in range(self.tile_sublayers):
            self._apply_cnot_pairs(tile_pairs)

    def _apply_no_entanglement_layer(self, layer_idx: int, weights):
        del layer_idx
        self._apply_rot_wall(weights)

    def _build_no_data_reupload_circuit(self, apply_layer, index_provider=None):
        @qml.qnode(self.device)
        def circuit(x):
            self._angle_embed(x)
            for layer in range(self.n_layers):
                idx = layer if index_provider is None else index_provider(layer)
                apply_layer(idx, self._main_layer_weights(layer))
            return qml.state()

        return circuit

    def _build_data_reupload_circuit(self, apply_layer, index_provider=None):
        @qml.qnode(self.device)
        def circuit(x):
            for layer in range(self.n_layers):
                for rep in range(self.data_reupload_every):
                    idx = rep if index_provider is None else index_provider(layer, rep)
                    apply_layer(idx, self._data_re_layer_weights(layer, rep))
                self._angle_embed(x)

            for rep in range(self.data_reupload_every):
                idx = rep if index_provider is None else index_provider(self.n_layers, rep)
                apply_layer(idx, self._last_data_re_weights(rep))

            return qml.state()

        return circuit

    def measure_state(self, state):
        return tq.measure(
            state,
            self.observables,
            pauli_chunk_size=self.pauli_measurement_chunk_size,
        )

    def circuit_basic_entangling(self):
        return self._build_no_data_reupload_circuit(self._apply_basic_entangling_layer)

    def data_re_circuit_basic_entangling(self):
        return self._build_data_reupload_circuit(self._apply_basic_entangling_layer)

    def circuit_single_rot_basic_ent(self):
        return self._build_no_data_reupload_circuit(self._apply_single_rot_basic_ent_layer)

    def data_re_circuit_single_rot_basic_ent(self):
        return self._build_data_reupload_circuit(self._apply_single_rot_basic_ent_layer)

    def circuit_strongly_entangling(self):
        return self._build_no_data_reupload_circuit(
            self._apply_strongly_entangling_layer,
            index_provider=lambda layer: layer,
        )

    def data_re_circuit_strongly_entangling(self):
        return self._build_data_reupload_circuit(
            self._apply_strongly_entangling_layer,
            index_provider=lambda _layer, rep: rep,
        )

    def circuit_cross_mesh(self):
        return self._build_no_data_reupload_circuit(self._apply_cross_mesh_layer)

    def data_re_circuit_cross_mesh(self):
        return self._build_data_reupload_circuit(self._apply_cross_mesh_layer)

    def circuit_cross_mesh_2_rots(self):
        return self._build_no_data_reupload_circuit(self._apply_cross_mesh_2_rots_layer)

    def data_re_circuit_cross_mesh_2_rots(self):
        return self._build_data_reupload_circuit(self._apply_cross_mesh_2_rots_layer)

    def circuit_cross_mesh_cx_rot(self):
        return self._build_no_data_reupload_circuit(self._apply_cross_mesh_cx_rot_layer)

    def data_re_circuit_cross_mesh_cx_rot(self):
        return self._build_data_reupload_circuit(self._apply_cross_mesh_cx_rot_layer)

    def circuit_tile(self):
        return self._build_no_data_reupload_circuit(self._apply_tile_layer)

    def data_re_circuit_tile(self):
        return self._build_data_reupload_circuit(self._apply_tile_layer)

    def circuit_no_entanglement_ansatz(self):
        return self._build_no_data_reupload_circuit(self._apply_no_entanglement_layer)

    def data_re_circuit_no_entanglement_ansatz(self):
        return self._build_data_reupload_circuit(self._apply_no_entanglement_layer)

    def circuit_no_entanglement(self):
        return self.circuit_no_entanglement_ansatz()

    def data_re_circuit_no_entanglement(self):
        return self.data_re_circuit_no_entanglement_ansatz()

    def qinr_circuit(self):
        weights1 = self._params_no_data_reupload()
        weights2 = self.params_last_layer_reupload
        imprimitive = qml.ops.CZ

        @qml.qnode(self.device)
        def circuit(x):
            for layer in range(self.n_layers):
                qml.StronglyEntanglingLayers(
                    weights1[layer].unsqueeze(0),
                    wires=range(self.n_qubits),
                    imprimitive=imprimitive,
                )
                for qubit in range(self.n_qubits):
                    qml.RZ(x[qubit], wires=qubit)
            if weights2 is not None:
                qml.StronglyEntanglingLayers(
                    weights2,
                    wires=range(self.n_qubits),
                    imprimitive=imprimitive,
                )
            return [qml.expval(qml.PauliZ(qubit)) for qubit in range(self.n_qubits)]

        return circuit


def _resolve_per_layer_param_shape(ansatz_name, n_qubits, n_layers, config=None):
    from torq.Ansatz import make_ansatz

    cfg = config if config is not None else SimpleNamespace()
    ansatz = make_ansatz(ansatz_name, n_qubits, n_layers, device=None, config=cfg)
    return tuple(n_qubits if dim is None else dim for dim in ansatz.per_layer_param_shape)


def get_input_and_weights(ansatz, n_qubits=5, n_layers=3, config=None):
    per_layer_shape = _resolve_per_layer_param_shape(ansatz, n_qubits, n_layers, config=config)
    x = torch.rand(n_qubits)
    weights = torch.rand(n_layers, 1, *per_layer_shape)
    return x, weights, None


def get_input_and_weights_data_re(
    ansatz,
    n_qubits=5,
    n_layers=3,
    data_reupload_every=0,
    config=None,
):
    per_layer_shape = _resolve_per_layer_param_shape(ansatz, n_qubits, n_layers, config=config)
    x = torch.rand(n_qubits)
    weights = torch.rand(n_layers, data_reupload_every, *per_layer_shape)
    weights_last_layer_data_re = torch.rand(data_reupload_every, *per_layer_shape)
    return x, weights, weights_last_layer_data_re


def _select_demo_circuit(comparison, ansatz_name, data_reupload_every):
    method_map = _DATA_RE_CIRCUIT_METHODS if data_reupload_every else _NO_DATA_CIRCUIT_METHODS
    method_name = method_map.get(ansatz_name)
    if method_name is None or not hasattr(comparison, method_name):
        raise ValueError(
            f"PennyLaneComparison does not support ansatz_name={ansatz_name!r} "
            f"with data_reupload_every={data_reupload_every}."
        )
    return getattr(comparison, method_name)()


def run_and_draw_circ(
    ansatz,
    circuit_factory_or_title,
    title=None,
    n_qubits=5,
    n_layers=3,
    data_reupload_every=0,
    config=None,
):
    if data_reupload_every == 0:
        x, weights, weights_last_layer_data_re = get_input_and_weights(
            ansatz,
            n_qubits,
            n_layers,
            config=config,
        )
    else:
        x, weights, weights_last_layer_data_re = get_input_and_weights_data_re(
            ansatz,
            n_qubits,
            n_layers,
            data_reupload_every,
            config=config,
        )

    comparison = PennyLaneComparison(
        n_qubits=n_qubits,
        n_layers=n_layers,
        weights=weights,
        weights_last_layer_data_re=weights_last_layer_data_re,
        data_reupload_every=data_reupload_every,
        config=config,
    )
    if callable(circuit_factory_or_title):
        circuit = circuit_factory_or_title(comparison)
        resolved_title = title if title is not None else ansatz
    else:
        circuit = _select_demo_circuit(comparison, ansatz, data_reupload_every)
        resolved_title = circuit_factory_or_title if title is None else title
    fig, _ax = qml.draw_mpl(circuit, level="device")(x)
    fig.suptitle(resolved_title, fontsize="xx-large")


# Backward compatibility for TorQ and older integrations.
qml_sanity_check = PennyLaneComparison
parent_module = sys.modules.get(__package__)
if parent_module is not None:
    setattr(parent_module, "PennyLaneComparison", PennyLaneComparison)


if __name__ == "__main__":
    torch.manual_seed(0)
    n_qubits = 5
    n_layers = 3
    data_reupload_every = 2

    demo_cases = (
        ("basic_entangling", None, "basic_entangling"),
        ("single_rot_basic_ent", SimpleNamespace(single_rotation_gate="ry"), "single_rot_basic_ent (RY)"),
        ("strongly_entangling", None, "strongly_entangling"),
        ("cross_mesh", None, "cross_mesh"),
        ("cross_mesh_2_rots", None, "cross_mesh_2_rots"),
        ("cross_mesh_cx_rot", None, "cross_mesh_cx_rot"),
        ("tile", None, "tile"),
        (
            "tile",
            SimpleNamespace(tile_rotation_params=1, single_rotation_gate="rz", tile_sublayers=2, tile_cyclic=True),
            "tile (1-param RZ, 2 sublayers, cyclic)",
        ),
        ("no_entanglement_ansatz", None, "no_entanglement_ansatz"),
    )

    for ansatz_name, config, title in demo_cases:
        run_and_draw_circ(
            ansatz_name,
            f"{title} - data re-upload",
            n_qubits=n_qubits,
            n_layers=n_layers,
            data_reupload_every=data_reupload_every,
            config=config,
        )

    for ansatz_name, config, title in demo_cases:
        run_and_draw_circ(
            ansatz_name,
            f"{title} - no data re-upload",
            n_qubits=n_qubits,
            n_layers=n_layers,
            config=config,
        )
