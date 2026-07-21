import torch

from pignn_glof.attribution import exact_modality_shapley
from pignn_glof.data.graph import build_graph
from pignn_glof.data.loader import collate_graph_samples
from pignn_glof.data.schema import GraphSample
from pignn_glof.models import PIGNN
from pignn_glof.physics import physics_losses


def sample(nodes: int = 8, horizon: int = 3):
    torch.manual_seed(3)
    coords = torch.rand(nodes, 2) * 1000
    elevation = 5000 - coords[:, 1] * 0.1
    flow = torch.tensor([[0.0, 1.0]]).repeat(nodes, 1)
    edge_index, edge_attr = build_graph(coords, elevation, flow, top_k=3)
    return GraphSample(
        optical=torch.rand(nodes, 6, 12, 12),
        velocity=torch.rand(nodes, 5, 3),
        thermal=torch.rand(nodes, 5, 1),
        coords=coords,
        elevation=elevation,
        flow=flow,
        labels=torch.randint(0, 2, (nodes,)).float(),
        trajectory=torch.rand(nodes, horizon),
        edge_index=edge_index,
        edge_attr=edge_attr,
        region_id="test-region",
        timestamp="2025-01-01",
    )


def test_model_shapes_physics_gradient_and_shapley_efficiency():
    batch = collate_graph_samples([sample()])
    model = PIGNN(
        modality_embedding=12,
        hidden_dim=24,
        graph_layers=2,
        forecast_horizon=3,
        dropout=0.0,
    )
    model.eval()
    output = model(batch)
    assert output["hazard_logits"].shape == (8,)
    assert output["trajectory_logits"].shape == (8, 3)
    losses = physics_losses(
        output["hazard_probability"],
        batch,
        {
            "continuity_weight": 0.5,
            "temperature_elevation_weight": 0.3,
            "flow_weight": 0.2,
            "downstream_margin": 0.02,
        },
    )
    assert torch.isfinite(losses.total)
    (output["hazard_logits"].mean() + losses.total).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    shapley = exact_modality_shapley(model, batch)
    assert shapley["values"].shape == (8, 3)
    assert float(shapley["efficiency_residual"].abs().max()) < 1e-5
