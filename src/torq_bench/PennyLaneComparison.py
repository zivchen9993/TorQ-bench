try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError("Install with `pip install torq-bench[pennylane]`") from exc
import torch
import numpy as np

class qml_sanity_check:
    def __init__(self, n_qubits=3, n_layers=1, weights=None, weights_last_layer_data_re=None,
                 data_reupload_every=0, pennylane_dev_name="default.qubit"):
        self.device    = qml.device(pennylane_dev_name, wires=n_qubits)
        # self.device    = qml.device("lightning.qubit", wires=n_qubits)
        # self.device    = qml.device("lightning.gpu", wires=n_qubits)
        # self.device    = qml.device("lightning.kokkos", wires=n_qubits)
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.params    = weights
        self.params_last_layer_reupload = weights_last_layer_data_re
        self.data_reupload_every = data_reupload_every
        print(f"[qml_sanity_check] Initialized with {n_qubits} qubits and {n_layers} layers.")

        #### QIREN ####
        self.in_features = n_qubits  # to keep the naming

    # 1) basic‑entangling (SX‑based mixer + wrap‑around ladder)
    def circuit_basic_entangling(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L = self.n_qubits, self.n_layers
            qml.AngleEmbedding(x, wires=range(n))

            ranges = L * [1]
            qml.StronglyEntanglingLayers(self.params.squeeze(), range(n), ranges=ranges)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 2) Strongly‑entangling
    def circuit_strongly_entangling(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L = self.n_qubits, self.n_layers
            qml.AngleEmbedding(x, wires=range(n))

            qml.StronglyEntanglingLayers(self.params.squeeze(), range(n))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 3) Cross‑mesh (single RX + CRZ)
    def circuit_cross_mesh(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L = self.n_qubits, self.n_layers
            qml.AngleEmbedding(x, wires=range(n))
            pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]

            for layer in range(L):
                w = self.params[layer].flatten()   # shape [n**2]
                # 1) single RX
                for q in range(n):
                    qml.RX(w[q], wires=q)
                # 2) cross‑mesh CRZs
                for idx,(c,t) in enumerate(pairs):
                    qml.CRZ(w[n + idx], wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 4) Cross‑mesh with 2 rotations per qubit (RX then RZ)
    def circuit_cross_mesh_2_rots(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L = self.n_qubits, self.n_layers
            qml.AngleEmbedding(x, wires=range(n))
            pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]
            for layer in range(L):
                w = self.params[layer].flatten()   # shape [2*n + n*(n-1)]
                theta1 = w[:n]
                theta2 = w[n:2*n]
                # first RX then RZ per qubit
                for q in range(n):
                    qml.RX(theta1[q], wires=q)
                    qml.RZ(theta2[q], wires=q)
                # cross‑mesh CRZ
                for idx,(c,t) in enumerate(pairs):
                    qml.CRZ(w[2*n + idx], wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 5) Cross‑mesh controlled‑X with composite “Rot” gates first
    def circuit_cross_mesh_cx_rot(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L = self.n_qubits, self.n_layers
            qml.AngleEmbedding(x, wires=range(n))
            pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]

            for layer in range(L):
                w = self.params[layer]  # shape [n,3]
                # 1) composite Rot on each qubit
                for q in range(n):
                    qml.Rot(w[q,0], w[q,1], w[q,2], wires=q)
                # 2) CNOT on every distinct pair
                for c,t in pairs:
                    qml.CNOT(wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit


    #### data reuploading checks ####
    # 1) basic‑entangling (SX‑based mixer + wrap‑around ladder)
    def data_re_circuit_basic_entangling(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L, K = self.n_qubits, self.n_layers, self.data_reupload_every
            ranges = K * [1]

            for layer in range(L):
                w = self.params[layer]      # shape [n,3]
                qml.StronglyEntanglingLayers(w.unsqueeze(0), wires=range(n), ranges=ranges)
                qml.AngleEmbedding(x, wires=range(n))

            # last layer
            qml.StronglyEntanglingLayers(self.params_last_layer_reupload.unsqueeze(0), wires=range(n), ranges=ranges)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 2) Strongly‑entangling
    def data_re_circuit_strongly_entangling(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L, K = self.n_qubits, self.n_layers, self.data_reupload_every

            for layer in range(L):
                w = self.params[layer]      # shape [n,3]
                qml.StronglyEntanglingLayers(w.unsqueeze(0), wires=range(n))
                qml.AngleEmbedding(x, wires=range(n))

            # last layer
            qml.StronglyEntanglingLayers(self.params_last_layer_reupload.unsqueeze(0), wires=range(n))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 3) Cross‑mesh (single RX + CRZ)
    def data_re_circuit_cross_mesh(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L, K = self.n_qubits, self.n_layers, self.data_reupload_every
            pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]

            for layer in range(L):
                for k in range(K):
                    w = self.params[layer, k].flatten()   # shape [n**2]
                    # 1) single RX
                    for q in range(n):
                        qml.RX(w[q], wires=q)
                    # 2) cross‑mesh CRZs
                    for idx,(c,t) in enumerate(pairs):
                        qml.CRZ(w[n + idx], wires=[c, t])
                qml.AngleEmbedding(x, wires=range(n))

            # last layer
            for k in range(K):
                w = self.params_last_layer_reupload[k].flatten()  # shape [n**2]
                # 1) single RX
                for q in range(n):
                    qml.RX(w[q], wires=q)
                # 2) cross‑mesh CRZs
                for idx, (c, t) in enumerate(pairs):
                    qml.CRZ(w[n + idx], wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 4) Cross‑mesh with 2 rotations per qubit (RX then RZ)
    def data_re_circuit_cross_mesh_2_rots(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L, K = self.n_qubits, self.n_layers, self.data_reupload_every

            pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]
            for layer in range(L):
                for k in range(K):
                    w = self.params[layer, k].flatten()   # shape [2*n + n*(n-1)]
                    theta1 = w[:n]
                    theta2 = w[n:2*n]
                    # first RX then RZ per qubit
                    for q in range(n):
                        qml.RX(theta1[q], wires=q)
                        qml.RZ(theta2[q], wires=q)
                    # cross‑mesh CRZ
                    for idx,(c,t) in enumerate(pairs):
                        qml.CRZ(w[2*n + idx], wires=[c, t])
                qml.AngleEmbedding(x, wires=range(n))

            # last layer
            for k in range(K):
                w = self.params_last_layer_reupload[k].flatten()  # shape [2*n + n*(n-1)]
                theta1 = w[:n]
                theta2 = w[n:2 * n]
                # first RX then RZ per qubit
                for q in range(n):
                    qml.RX(theta1[q], wires=q)
                    qml.RZ(theta2[q], wires=q)
                # cross‑mesh CRZ
                for idx, (c, t) in enumerate(pairs):
                    qml.CRZ(w[2 * n + idx], wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    # 5) Cross‑mesh controlled‑X with composite “Rot” gates first
    def data_re_circuit_cross_mesh_cx_rot(self):
        @qml.qnode(self.device)
        def circuit(x):
            n, L, K = self.n_qubits, self.n_layers, self.data_reupload_every

            for layer in range(L):
                for k in range(K):
                    w = self.params[layer, k]  # shape [n,3]
                    # 1) composite Rot on each qubit
                    for q in range(n):
                        qml.Rot(w[q,0], w[q,1], w[q,2], wires=q)
                    # 2) CNOT on every distinct pair
                    pairs = [(c,t) for c in reversed(range(n)) for t in reversed(range(n)) if c!=t]
                    for c,t in pairs:
                        qml.CNOT(wires=[c, t])
                qml.AngleEmbedding(x, wires=range(n))

            # last layer
            for k in range(K):
                w = self.params_last_layer_reupload[k]  # shape [n,3]
                # 1) composite Rot on each qubit
                for q in range(n):
                    qml.Rot(w[q, 0], w[q, 1], w[q, 2], wires=q)
                # 2) CNOT on every distinct pair
                pairs = [(c, t) for c in reversed(range(n)) for t in reversed(range(n)) if c != t]
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n)]
        return circuit

    def qinr_circuit(self):
        weights1 = self.params
        weights2 = self.params_last_layer_reupload
        imprimitive = qml.ops.CZ
        # imprimitive = qml.ops.CNOT
        @qml.qnode(self.device)
        def circuit(x):
            for i in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=imprimitive)
                for j in range(self.in_features):
                    qml.RZ(x[j], wires=j)
            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=imprimitive)

            # if self.use_noise != 0:
            #     for i in range(self.in_features):
            #         rand_angle = np.pi + self.use_noise * np.random.rand()
            #         qml.RX(rand_angle, wires=i)

            res = []
            for i in range(self.in_features):
                res.append(qml.expval(qml.PauliZ(i)))
            return res

        return circuit

def run_and_draw_circ(ansatz, circuit_factory, title, n_qubits=5, n_layers=3, data_reupload_every=0):
    if data_reupload_every == 0:
        x, weights, weights_last_layer_data_re = get_input_and_weights(ansatz, n_qubits, n_layers)
    else:
        x, weights, weights_last_layer_data_re = get_input_and_weights_data_re(ansatz, n_qubits, n_layers, data_reupload_every)
    qml_sanity_check_class = qml_sanity_check(n_qubits=n_qubits, n_layers=n_layers,
                                              weights=weights, weights_last_layer_data_re=weights_last_layer_data_re,
                                              data_reupload_every=data_reupload_every)
    circuit = circuit_factory(qml_sanity_check_class)
    # print("title: ", qml.draw(circuit)(x))
    fig, ax = qml.draw_mpl(circuit, level="device")(x)
    fig.suptitle(title, fontsize="xx-large")
    # fig.show()

def get_input_and_weights(ansatz, n_qubits=5, n_layers=3):  # data_reupload_every=0 is regular without datareupload
    if ansatz == "cross_mesh":
        weights = torch.rand(n_layers, n_qubits ** 2)
    elif ansatz == "cross_mesh_2_rots":
        weights = torch.rand(n_layers, n_qubits + n_qubits ** 2)
    else:
        weights = torch.rand(n_layers, n_qubits, 3)

    x = torch.rand(n_qubits)
    return x, weights, None  # weights_last_layer_data_re is None in this case


def get_input_and_weights_data_re(ansatz, n_qubits=5, n_layers=3, data_reupload_every=0):  # data_reupload_every=0 is regular without datareupload
    if ansatz == "cross_mesh":
        weights = torch.rand(n_layers, data_reupload_every, n_qubits ** 2)
        weights_last_layer_data_re = torch.rand(data_reupload_every, n_qubits ** 2)
    elif ansatz == "cross_mesh_2_rots":
        weights = torch.rand(n_layers, data_reupload_every, n_qubits + n_qubits ** 2)
        weights_last_layer_data_re = torch.rand(data_reupload_every, n_qubits + n_qubits ** 2)
    else:
        weights = torch.rand(n_layers, data_reupload_every, n_qubits, 3)
        weights_last_layer_data_re = torch.rand(data_reupload_every, n_qubits, 3)

    x = torch.rand(n_qubits)
    return x, weights, weights_last_layer_data_re

if __name__ == "__main__":
    torch.manual_seed(0)
    n_qubits = 5
    n_layers = 3
    data_reupload_every = 2

    ANSATZ_CONFIGS_DATA_RE = [
        ("basic_entangling", lambda qc: qc.data_re_circuit_basic_entangling(), "data_re_circuit_basic_entangling"),
        ("strongly_entangling", lambda qc: qc.data_re_circuit_strongly_entangling(), "data_re_circuit_strongly_entangling"),
        ("cross_mesh", lambda qc: qc.data_re_circuit_cross_mesh(), "data_re_circuit_cross_mesh"),
        ("cross_mesh_2_rots", lambda qc: qc.data_re_circuit_cross_mesh_2_rots(), "data_re_circuit_cross_mesh_2_rots"),
        ("cross_mesh_cx_rot", lambda qc: qc.data_re_circuit_cross_mesh_cx_rot(), "data_re_circuit_cross_mesh_cx_rot"),
        ("strongly_entangling", lambda qc: qc.qinr_circuit(), "qinr_circuit"),
    ]
    ANSATZ_CONFIGS = [
        ("basic_entangling", lambda qc: qc.circuit_basic_entangling(), "circuit_basic_entangling - no data-re"),
        ("strongly_entangling", lambda qc: qc.circuit_strongly_entangling(), "circuit_strongly_entangling - no data-re"),
        ("cross_mesh", lambda qc: qc.circuit_cross_mesh(), "circuit_cross_mesh - no data-re"),
        ("cross_mesh_2_rots", lambda qc: qc.circuit_cross_mesh_2_rots(), "circuit_cross_mesh_2_rots - no data-re"),
        ("cross_mesh_cx_rot", lambda qc: qc.circuit_cross_mesh_cx_rot(), "circuit_cross_mesh_cx_rot - no data-re"),
    ]

    # with data-reupload
    run_and_draw_circ("basic_entangling", lambda qc: qc.data_re_circuit_basic_entangling(), "data_re_circuit_basic_entangling", n_qubits, n_layers, data_reupload_every)
    run_and_draw_circ("strongly_entangling", lambda qc: qc.data_re_circuit_strongly_entangling(), "data_re_circuit_strongly_entangling", n_qubits, n_layers, data_reupload_every)
    run_and_draw_circ("cross_mesh", lambda qc: qc.data_re_circuit_cross_mesh(), "data_re_circuit_cross_mesh", n_qubits, n_layers, data_reupload_every)
    run_and_draw_circ("cross_mesh_2_rots", lambda qc: qc.data_re_circuit_cross_mesh_2_rots(), "data_re_circuit_cross_mesh_2_rots", n_qubits, n_layers, data_reupload_every)
    run_and_draw_circ("cross_mesh_cx_rot", lambda qc: qc.data_re_circuit_cross_mesh_cx_rot(), "data_re_circuit_cross_mesh_cx_rot", n_qubits, n_layers, data_reupload_every)
    run_and_draw_circ("strongly_entangling", lambda qc: qc.qinr_circuit(), "qinr_circuit", n_qubits, n_layers, data_reupload_every)

    # without data-reupload
    run_and_draw_circ("basic_entangling", lambda qc: qc.circuit_basic_entangling(), "circuit_basic_entangling - no data-re", n_qubits, n_layers)
    run_and_draw_circ("strongly_entangling", lambda qc: qc.circuit_strongly_entangling(), "circuit_strongly_entangling - no data-re", n_qubits, n_layers)
    run_and_draw_circ("cross_mesh", lambda qc: qc.circuit_cross_mesh(), "circuit_cross_mesh - no data-re", n_qubits, n_layers)
    run_and_draw_circ("cross_mesh_2_rots", lambda qc: qc.circuit_cross_mesh_2_rots(), "circuit_cross_mesh_2_rots - no data-re", n_qubits, n_layers)
    run_and_draw_circ("cross_mesh_cx_rot", lambda qc: qc.circuit_cross_mesh_cx_rot(), "circuit_cross_mesh_cx_rot - no data-re", n_qubits, n_layers)
