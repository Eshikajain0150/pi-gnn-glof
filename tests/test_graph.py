import torch

from pignn_glof.data.graph import EDGE_DOWNHILL, EDGE_WEIGHT, build_graph


def test_directed_graph_has_no_self_edges_and_normalized_outgoing_weights():
    coords = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    elevation = torch.tensor([4.0, 3.0, 2.0, 3.0])
    flow = torch.tensor([[0.0, 1.0]]).repeat(4, 1)
    edge_index, edge_attr = build_graph(coords, elevation, flow, top_k=2)
    src, dst = edge_index
    assert torch.all(src != dst)
    sums = torch.zeros(coords.shape[0])
    sums.index_add_(0, src, edge_attr[:, EDGE_WEIGHT])
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(edge_attr[:, EDGE_DOWNHILL] >= 0)
