import pytest


def test_pennylane_comparison_export_and_alias():
    pytest.importorskip("pennylane")

    from torq_bench import PennyLaneComparison as ExportedComparison
    from torq_bench.PennyLaneComparison import PennyLaneComparison, qml_sanity_check

    assert ExportedComparison is PennyLaneComparison
    assert qml_sanity_check is PennyLaneComparison


def test_pennylane_q_layer_can_be_constructed():
    pytest.importorskip("pennylane")
    pytest.importorskip("torq")

    import torch
    from torq_bench.pennylane_backend import PennyLaneQLayer

    layer = PennyLaneQLayer(n_qubits=2, n_layers=1, ansatz_name="basic_entangling")
    out = layer(torch.rand(3, 2))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
