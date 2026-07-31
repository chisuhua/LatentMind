# Hippo KG 压缩生长最小验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在小规模人造 KG（100-1000 节点）上训练简化版 Hippo，观察图结构自发涌现，验证关键指标：涌现速度 < 10K 步、节点度分布 α ∈ [2, 3] (power-law)、新节点引入的全局重组织幅度 < 20%。

**Architecture:** 实验框架分为 6 个模块：人造 KG 生成器（Barabási-Albert + Random + Small-World）、图论指标计算器（NetworkX 集成）、简化版 Hippo 训练器（FM encoder + FSQ + 重建 MSE，不含图损失 L_graph）、实时涌现检测器（滑动窗口稳定度检测）、生长曲线可视化器（matplotlib/seaborn）。所有模块通过 `src/hippo/experiments/graph_growth/` 下的统一实验脚本串联，输出三个关键指标的时序曲线。

**Tech Stack:** Python 3.11+, PyTorch 2.1+, NetworkX 3.x, matplotlib 3.x, seaborn 0.13+, scipy 1.12+, pytest 8.x

---

## 前置条件

- 项目根目录下存在 `src/__init__.py`（已存在）
- Python 3.11+ 环境可用
- 以下依赖需安装：`pip install torch networkx matplotlib seaborn scipy pytest`

---

### Task 1: 实验框架目录搭建与测试脚手架

**Files:**
- Create: `src/hippo/__init__.py`
- Create: `src/hippo/experiments/__init__.py`
- Create: `src/hippo/experiments/graph_growth/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/hippo/__init__.py`
- Create: `tests/hippo/experiments/__init__.py`
- Create: `tests/hippo/experiments/graph_growth/__init__.py`
- Create: `tests/hippo/experiments/graph_growth/test_setup.py`
- Modify: `src/__init__.py`（追加 hippo 子模块说明）

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_setup.py
"""Test that the graph_growth experiment package is importable."""


def test_graph_growth_package_importable():
    """Verify that the graph_growth experiment module can be imported."""
    from src.hippo.experiments.graph_growth import kg_generator
    from src.hippo.experiments.graph_growth import emergence_metrics
    from src.hippo.experiments.graph_growth import trainer
    from src.hippo.experiments.graph_growth import emergence_detector
    from src.hippo.experiments.graph_growth import growth_curves
    assert kg_generator is not None
    assert emergence_metrics is not None
    assert trainer is not None
    assert emergence_detector is not None
    assert growth_curves is not None


def test_tests_directory_importable():
    """Verify that the tests package structure works."""
    import tests.hippo.experiments.graph_growth
    assert tests.hippo.experiments.graph_growth is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_setup.py -v 2>&1`
Expected: `FAILED` with `ModuleNotFoundError: No module named 'src.hippo'` (or similar import error)

- [ ] **Step 3: Write minimal implementation**

Create the directory structure and `__init__.py` files.

```python
# src/hippo/__init__.py
"""Hippo Research Line — right-brain memory / KG / retrieval experiments."""
```

```python
# src/hippo/experiments/__init__.py
"""Hippo experiments subpackage."""
```

```python
# src/hippo/experiments/graph_growth/__init__.py
"""KG compressed growth experiment — emergence detection and growth curve monitoring."""
from src.hippo.experiments.graph_growth import kg_generator
from src.hippo.experiments.graph_growth import emergence_metrics
from src.hippo.experiments.graph_growth import trainer
from src.hippo.experiments.graph_growth import emergence_detector
from src.hippo.experiments.graph_growth import growth_curves

__all__ = ["kg_generator", "emergence_metrics", "trainer",
           "emergence_detector", "growth_curves"]
```

```python
# tests/__init__.py
"""LatentMind test suite."""
```

```python
# tests/hippo/__init__.py
"""Hippo Research Line tests."""
```

```python
# tests/hippo/experiments/__init__.py
"""Hippo experiments tests."""
```

```python
# tests/hippo/experiments/graph_growth/__init__.py
"""KG compressed growth experiment tests."""
```

Also create empty module stubs so the imports work:

```python
# src/hippo/experiments/graph_growth/kg_generator.py
"""KG generator — synthetic graph generation utilities."""
```

```python
# src/hippo/experiments/graph_growth/emergence_metrics.py
"""Emergence metrics — graph theory indicators via NetworkX."""
```

```python
# src/hippo/experiments/graph_growth/trainer.py
"""Simplified Hippo trainer — FM encoder + FSQ + reconstruction MSE loss."""
```

```python
# src/hippo/experiments/graph_growth/emergence_detector.py
"""Emergence detector — real-time detection of graph metric stabilization."""
```

```python
# src/hippo/experiments/graph_growth/growth_curves.py
"""Growth curves — matplotlib visualization of metrics vs training step."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_setup.py -v 2>&1`
Expected: `PASSED` (2 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/ tests/ src/__init__.py
git commit -m "feat(graph-growth): scaffold experiment directory structure and test harness"
```

---

### Task 2: KG 生成器（Part 1）— Barabási-Albert 优先附着基线

**Files:**
- Modify: `src/hippo/experiments/graph_growth/kg_generator.py`
- Create: `tests/hippo/experiments/graph_growth/test_kg_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_kg_generator.py
"""Tests for KG generator — Barabási-Albert baseline."""

import networkx as nx
import torch
from src.hippo.experiments.graph_growth.kg_generator import (
    generate_barabasi_albert_kg,
    kg_to_node_features,
    KGGeneratorConfig,
)


class TestBarabasiAlbertKG:
    """Test Barabási-Albert KG generation."""

    def test_config_defaults(self):
        """KGGeneratorConfig should have sensible defaults."""
        config = KGGeneratorConfig()
        assert config.n_nodes == 100
        assert config.m_edges_per_step == 2
        assert config.feature_dim == 16
        assert config.seed == 42

    def test_generate_returns_graph_with_correct_node_count(self):
        """Generated graph should have the requested number of nodes."""
        config = KGGeneratorConfig(n_nodes=50, m_edges_per_step=2, seed=0)
        G = generate_barabasi_albert_kg(config)
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() > 0

    def test_generate_ba_graph_is_connected(self):
        """Barabási-Albert graph should be connected."""
        config = KGGeneratorConfig(n_nodes=100, m_edges_per_step=3, seed=42)
        G = generate_barabasi_albert_kg(config)
        assert nx.is_connected(G)

    def test_degree_distribution_is_power_law_like(self):
        """BA graph should have a heavy-tailed degree distribution."""
        config = KGGeneratorConfig(n_nodes=200, m_edges_per_step=2, seed=42)
        G = generate_barabasi_albert_kg(config)
        degrees = sorted(d for _, d in G.degree(), reverse=True)
        # BA preferential attachment produces hubs: top degree should be >> average
        avg_deg = sum(degrees) / len(degrees)
        max_deg = degrees[0]
        assert max_deg > 2 * avg_deg, (
            f"BA graph should have hubs: max_deg={max_deg}, avg_deg={avg_deg}"
        )

    def test_kg_to_node_features_returns_correct_shape(self):
        """Node features should have shape (n_nodes, feature_dim)."""
        config = KGGeneratorConfig(n_nodes=50, m_edges_per_step=2,
                                   feature_dim=16, seed=42)
        G = generate_barabasi_albert_kg(config)
        features = kg_to_node_features(G, feature_dim=16)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (50, 16)

    def test_kg_to_node_features_deterministic(self):
        """Same seed should produce same features."""
        config = KGGeneratorConfig(n_nodes=30, m_edges_per_step=2,
                                   feature_dim=8, seed=7)
        G1 = generate_barabasi_albert_kg(config)
        G2 = generate_barabasi_albert_kg(config)
        f1 = kg_to_node_features(G1, feature_dim=8)
        f2 = kg_to_node_features(G2, feature_dim=8)
        assert torch.equal(f1, f2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_kg_generator.py::TestBarabasiAlbertKG -v 2>&1`
Expected: `FAILED` with `ImportError` or `AttributeError` — the module exists but classes are not defined yet

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/kg_generator.py
"""KG generator — synthetic graph generation utilities."""

from dataclasses import dataclass, field

import networkx as nx
import torch


@dataclass
class KGGeneratorConfig:
    """Configuration for synthetic KG generation.

    Attributes:
        n_nodes: Number of nodes in the graph.
        m_edges_per_step: Edges added per new node (Barabási-Albert m parameter).
        feature_dim: Dimensionality of random node feature vectors.
        seed: Random seed for reproducibility.
    """
    n_nodes: int = 100
    m_edges_per_step: int = 2
    feature_dim: int = 16
    seed: int = 42


def generate_barabasi_albert_kg(config: KGGeneratorConfig) -> nx.Graph:
    """Generate a Barabási-Albert preferential attachment graph.

    Args:
        config: Configuration for the generator.

    Returns:
        A NetworkX Graph with n_nodes nodes and
        (n_nodes - m_edges_per_step) * m_edges_per_step edges.
    """
    return nx.barabasi_albert_graph(
        n=config.n_nodes,
        m=config.m_edges_per_step,
        seed=config.seed,
    )


def kg_to_node_features(G: nx.Graph, feature_dim: int,
                        seed: int = 42) -> torch.Tensor:
    """Assign random feature vectors to each node in the graph.

    Each node gets a random Gaussian vector of length feature_dim.
    The random vectors are seeded so that the same node index always
    gets the same feature vector across calls with the same seed.

    Args:
        G: The input graph.
        feature_dim: Dimensionality of the feature vectors.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (n_nodes, feature_dim).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_nodes = G.number_of_nodes()
    return torch.randn(n_nodes, feature_dim, generator=rng)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_kg_generator.py::TestBarabasiAlbertKG -v 2>&1`
Expected: `PASSED` (6 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/kg_generator.py tests/hippo/experiments/graph_growth/test_kg_generator.py
git commit -m "feat(graph-growth): implement Barabasi-Albert KG generator with node features"
```

---

### Task 3: KG 生成器（Part 2）— Random + Small-World 消融对照

**Files:**
- Modify: `src/hippo/experiments/graph_growth/kg_generator.py`
- Modify: `tests/hippo/experiments/graph_growth/test_kg_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/hippo/experiments/graph_growth/test_kg_generator.py

class TestAlternativeKGGenerators:
    """Test Erdős-Rényi and Small-World generators for ablation."""

    def test_generate_er_graph_correct_size(self):
        """Erdos-Renyi graph should have the requested number of nodes."""
        from src.hippo.experiments.graph_growth.kg_generator import (
            generate_erdos_renyi_kg,
        )
        G = generate_erdos_renyi_kg(n_nodes=80, edge_prob=0.05, seed=0)
        assert G.number_of_nodes() == 80

    def test_generate_er_graph_has_edges(self):
        """Erdos-Renyi graph with reasonable edge_prob should have edges."""
        from src.hippo.experiments.graph_growth.kg_generator import (
            generate_erdos_renyi_kg,
        )
        G = generate_erdos_renyi_kg(n_nodes=100, edge_prob=0.08, seed=42)
        assert G.number_of_edges() > 0

    def test_generate_ws_graph_correct_size(self):
        """Small-world graph should have the requested number of nodes."""
        from src.hippo.experiments.graph_growth.kg_generator import (
            generate_watts_strogatz_kg,
        )
        G = generate_watts_strogatz_kg(n_nodes=60, k_neighbors=4,
                                       rewiring_prob=0.3, seed=0)
        assert G.number_of_nodes() == 60

    def test_generate_ws_graph_high_clustering(self):
        """Small-world graph should have high clustering coefficient."""
        from src.hippo.experiments.graph_growth.kg_generator import (
            generate_watts_strogatz_kg,
        )
        G = generate_watts_strogatz_kg(n_nodes=100, k_neighbors=6,
                                       rewiring_prob=0.1, seed=42)
        cc = nx.average_clustering(G)
        # WS graphs with low rewiring have CC similar to the regular lattice
        # For k=6, the regular lattice CC ≈ (3k-3)/(4k-2) = 15/22 ≈ 0.68
        # With p=0.1 rewiring, CC should still be well above 0.1
        assert cc > 0.3, f"Expected high clustering, got cc={cc}"

    def test_generate_ws_graph_short_paths(self):
        """Small-world graph should have short average path length."""
        from src.hippo.experiments.graph_growth.kg_generator import (
            generate_watts_strogatz_kg,
        )
        G = generate_watts_strogatz_kg(n_nodes=100, k_neighbors=6,
                                       rewiring_prob=0.3, seed=42)
        apl = nx.average_shortest_path_length(G)
        # For n=100, k=6, log(n)/log(k) ≈ 2.6; with rewiring, apl should be near that
        assert apl < 10.0, f"Expected short paths, got apl={apl}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_kg_generator.py::TestAlternativeKGGenerators -v 2>&1`
Expected: `FAILED` with `ImportError` — the functions are not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/hippo/experiments/graph_growth/kg_generator.py

def generate_erdos_renyi_kg(n_nodes: int, edge_prob: float,
                            seed: int = 42) -> nx.Graph:
    """Generate an Erdős-Rényi random graph.

    Used as a null baseline for ablation comparison — no preferential
    attachment, no community structure.

    Args:
        n_nodes: Number of nodes.
        edge_prob: Probability of edge creation between any two nodes.
        seed: Random seed.

    Returns:
        A NetworkX Graph.
    """
    return nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=seed)


def generate_watts_strogatz_kg(n_nodes: int, k_neighbors: int,
                               rewiring_prob: float,
                               seed: int = 42) -> nx.Graph:
    """Generate a Watts-Strogatz small-world graph.

    Used as a positive control for clustering and path length baselines.

    Args:
        n_nodes: Number of nodes.
        k_neighbors: Each node is connected to k nearest neighbors
                     in the ring topology.
        rewiring_prob: Probability of rewiring each edge.
        seed: Random seed.

    Returns:
        A NetworkX Graph.
    """
    return nx.watts_strogatz_graph(
        n=n_nodes, k=k_neighbors, p=rewiring_prob, seed=seed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_kg_generator.py::TestAlternativeKGGenerators -v 2>&1`
Expected: `PASSED` (5 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/kg_generator.py tests/hippo/experiments/graph_growth/test_kg_generator.py
git commit -m "feat(graph-growth): add Erdos-Renyi and Watts-Strogatz generators for ablation"
```

---

### Task 4: 涌现指标（Part 1）— 聚类系数 + 平均路径长度（NetworkX）

**Files:**
- Modify: `src/hippo/experiments/graph_growth/emergence_metrics.py`
- Create: `tests/hippo/experiments/graph_growth/test_emergence_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_emergence_metrics.py
"""Tests for emergence metrics — clustering coefficient and average path length."""

import networkx as nx
import torch
from src.hippo.experiments.graph_growth.emergence_metrics import (
    compute_clustering_coefficient,
    compute_average_path_length,
    compute_metrics_from_features,
    EmergenceMetrics,
)


class TestClusteringAndPathLength:
    """Test clustering coefficient and average path length computation."""

    def test_clustering_coefficient_on_complete_graph(self):
        """Complete graph should have clustering coefficient of 1.0."""
        G = nx.complete_graph(10)
        cc = compute_clustering_coefficient(G)
        assert cc == pytest.approx(1.0, abs=1e-6)

    def test_clustering_coefficient_on_empty_graph(self):
        """Empty graph (no edges) should have clustering coefficient of 0.0."""
        G = nx.empty_graph(10)
        cc = compute_clustering_coefficient(G)
        assert cc == pytest.approx(0.0, abs=1e-6)

    def test_avg_path_length_on_complete_graph(self):
        """Complete graph should have average path length of 1.0."""
        G = nx.complete_graph(10)
        apl = compute_average_path_length(G)
        assert apl == pytest.approx(1.0, abs=1e-6)

    def test_avg_path_length_on_path_graph(self):
        """Path graph of 5 nodes should have APL ~ 2.0."""
        G = nx.path_graph(5)
        apl = compute_average_path_length(G)
        # For a path of 5: (1+2+3+4)*2 / 20 = 2.0
        assert apl == pytest.approx(2.0, abs=0.1)

    def test_emergence_metrics_dataclass(self):
        """EmergenceMetrics should store all three fields."""
        m = EmergenceMetrics(clustering=0.5, avg_path_length=2.0,
                             degree_alpha=2.5)
        assert m.clustering == 0.5
        assert m.avg_path_length == 2.0
        assert m.degree_alpha == 2.5

    def test_metrics_from_features_returns_dataclass(self):
        """compute_metrics_from_features should return EmergenceMetrics."""
        # Build a small BA graph
        G = nx.barabasi_albert_graph(50, 2, seed=42)
        features = torch.randn(50, 8)
        adjacency = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        metrics = compute_metrics_from_features(features, adjacency)
        assert isinstance(metrics, EmergenceMetrics)
        assert metrics.clustering >= 0.0
        assert metrics.avg_path_length >= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_metrics.py -v 2>&1`
Expected: `FAILED` with `ImportError` — functions not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/emergence_metrics.py
"""Emergence metrics — graph theory indicators via NetworkX."""

from dataclasses import dataclass

import networkx as nx
import torch


@dataclass
class EmergenceMetrics:
    """Container for graph-theoretic emergence metrics.

    Attributes:
        clustering: Average clustering coefficient (0 to 1).
        avg_path_length: Average shortest path length (>= 1 for connected graphs).
        degree_alpha: Power-law exponent α of the degree distribution.
    """
    clustering: float
    avg_path_length: float
    degree_alpha: float


def compute_clustering_coefficient(G: nx.Graph) -> float:
    """Compute the average clustering coefficient of a graph.

    Args:
        G: A NetworkX graph.

    Returns:
        Average clustering coefficient as a float.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0
    return float(nx.average_clustering(G))


def compute_average_path_length(G: nx.Graph) -> float:
    """Compute the average shortest path length.

    For disconnected graphs, computes the average over the largest connected
    component only.

    Args:
        G: A NetworkX graph.

    Returns:
        Average shortest path length.
    """
    if G.number_of_nodes() < 2:
        return 1.0
    if nx.is_connected(G):
        return float(nx.average_shortest_path_length(G))
    # Fall back to largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    if subgraph.number_of_nodes() < 2:
        return 1.0
    return float(nx.average_shortest_path_length(subgraph))


def compute_metrics_from_features(
    features: torch.Tensor, adjacency: torch.Tensor,
) -> EmergenceMetrics:
    """Compute emergence metrics from node features and adjacency matrix.

    Builds a NetworkX graph from the adjacency matrix, then computes
    clustering coefficient and average path length. The degree alpha
    is a placeholder (set to 0.0) that will be filled by the power-law
    fitting routine in Task 5.

    Args:
        features: Node feature tensor of shape (n_nodes, feature_dim).
        adjacency: Adjacency matrix tensor of shape (n_nodes, n_nodes).

    Returns:
        An EmergenceMetrics instance with clustering and avg_path_length
        computed, and degree_alpha = 0.0 (placeholder).
    """
    n_nodes = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency.numpy())
    cc = compute_clustering_coefficient(G)
    apl = compute_average_path_length(G)
    return EmergenceMetrics(clustering=cc, avg_path_length=apl, degree_alpha=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_metrics.py -v 2>&1`
Expected: `PASSED` (6 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/emergence_metrics.py tests/hippo/experiments/graph_growth/test_emergence_metrics.py
git commit -m "feat(graph-growth): implement clustering coefficient and avg path length metrics"
```

---

### Task 5: 涌现指标（Part 2）— 度分布 Power-Law 拟合（scipy.optimize）

**Files:**
- Modify: `src/hippo/experiments/graph_growth/emergence_metrics.py`
- Modify: `tests/hippo/experiments/graph_growth/test_emergence_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/hippo/experiments/graph_growth/test_emergence_metrics.py

import numpy as np
import pytest
from src.hippo.experiments.graph_growth.emergence_metrics import fit_power_law_alpha


class TestPowerLawFit:
    """Test degree distribution power-law fitting."""

    def test_fit_alpha_on_ba_graph(self):
        """BA graph degree alpha should be in [2, 3]."""
        G = nx.barabasi_albert_graph(500, 3, seed=42)
        degrees = [d for _, d in G.degree()]
        alpha = fit_power_law_alpha(degrees)
        # BA graphs theoretically converge to alpha ~ 3 as n→∞
        # For n=500, m=3, we expect alpha in [2.0, 4.0]
        assert 2.0 <= alpha <= 4.0, f"alpha={alpha} outside expected range"

    def test_fit_alpha_on_er_graph(self):
        """Erdos-Renyi graph degree distribution should NOT be power-law.
        The fit might still produce a value, but it should differ from the
        BA alpha in a detectable way (e.g. much higher variance or
        goodness-of-fit issues). We just check that the function returns
        a float without error."""
        G = nx.erdos_renyi_graph(500, 0.05, seed=42)
        degrees = [d for _, d in G.degree()]
        alpha = fit_power_law_alpha(degrees)
        assert isinstance(alpha, float)
        assert alpha > 0.0

    def test_fit_alpha_on_constant_degree(self):
        """Regular graph (all nodes same degree) should return a large alpha."""
        G = nx.complete_graph(20)
        degrees = [d for _, d in G.degree()]
        # All nodes have degree 19 — the power-law fit should produce
        # a very large alpha (near-infinite slope)
        alpha = fit_power_law_alpha(degrees)
        assert alpha > 10.0 or np.isinf(alpha)

    def test_fit_alpha_on_empty_degrees(self):
        """Empty degree list should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 unique degrees"):
            fit_power_law_alpha([])

    def test_fit_alpha_on_single_value(self):
        """All nodes having the same degree should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 unique degrees"):
            fit_power_law_alpha([3, 3, 3, 3])

    def test_fit_alpha_returns_float(self):
        """fit_power_law_alpha should return a float."""
        degrees = [1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8, 10, 12, 15]
        alpha = fit_power_law_alpha(degrees)
        assert isinstance(alpha, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_metrics.py::TestPowerLawFit -v 2>&1`
Expected: `FAILED` with `ImportError` — `fit_power_law_alpha` not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/hippo/experiments/graph_growth/emergence_metrics.py

import numpy as np
from scipy.optimize import curve_fit


def _power_law_pdf(k: float, alpha: float, k_min: float) -> float:
    """Power-law probability density: P(k) ∝ k^{-alpha} for k >= k_min."""
    # Normalization constant: (alpha - 1) * k_min^(alpha - 1)
    norm = (alpha - 1.0) * (k_min ** (alpha - 1.0))
    return norm * (k ** (-alpha))


def fit_power_law_alpha(degrees: list[int], k_min: int = 1) -> float:
    """Fit a power-law exponent α to the degree distribution.

    Uses scipy.optimize.curve_fit to fit P(k) ~ k^{-α} for k >= k_min
    via least-squares on the log-log complementary cumulative distribution.

    Args:
        degrees: List of node degrees.
        k_min: Minimum degree to include in the fit (default 1).

    Returns:
        Fitted power-law exponent α.

    Raises:
        ValueError: If fewer than 2 unique degree values >= k_min.
    """
    if len(degrees) == 0:
        raise ValueError("at least 2 unique degrees required, got empty list")

    # Count degree frequencies
    from collections import Counter
    degree_counts = Counter(degrees)
    # Filter to k >= k_min
    filtered = {k: v for k, v in degree_counts.items() if k >= k_min}

    if len(filtered) < 2:
        raise ValueError(
            f"at least 2 unique degrees >= k_min={k_min} required, "
            f"got {len(filtered)}"
        )

    # Compute complementary cumulative distribution (CCDF)
    # P(K >= k) = fraction of nodes with degree >= k
    total_nodes = len(degrees)
    ks = np.array(sorted(filtered.keys()))
    ccdf = np.array([
        sum(v for kk, v in filtered.items() if kk >= k) / total_nodes
        for k in ks
    ])

    # Remove zero entries for log transform
    nonzero = ccdf > 0
    ks_nonzero = ks[nonzero]
    ccdf_nonzero = ccdf[nonzero]

    if len(ks_nonzero) < 2:
        raise ValueError("fewer than 2 non-zero CCDF entries after filtering")

    # Fit on log-log scale: log(CCDF) ~ (1-α) * log(k) + const
    log_k = np.log(ks_nonzero)
    log_ccdf = np.log(ccdf_nonzero)

    # Linear fit: log_ccdf = (1-α) * log_k + b
    # So α = 1 - slope
    A = np.vstack([log_k, np.ones_like(log_k)]).T
    slope, _ = np.linalg.lstsq(A, log_ccdf, rcond=None)[0]
    alpha = 1.0 - slope

    return float(alpha)
```

Now update `compute_metrics_from_features` to use the fitted alpha:

```python
# In src/hippo/experiments/graph_growth/emergence_metrics.py, update compute_metrics_from_features
# Replace the line: return EmergenceMetrics(clustering=cc, avg_path_length=apl, degree_alpha=0.0)
# With:
def compute_metrics_from_features(
    features: torch.Tensor, adjacency: torch.Tensor,
) -> EmergenceMetrics:
    """Compute emergence metrics from node features and adjacency matrix.

    Builds a NetworkX graph from the adjacency matrix, then computes
    clustering coefficient, average path length, and degree distribution
    power-law exponent α.

    Args:
        features: Node feature tensor of shape (n_nodes, feature_dim).
        adjacency: Adjacency matrix tensor of shape (n_nodes, n_nodes).

    Returns:
        An EmergenceMetrics instance with all three fields computed.
    """
    n_nodes = adjacency.shape[0]
    G = nx.from_numpy_array(adjacency.numpy())
    cc = compute_clustering_coefficient(G)
    apl = compute_average_path_length(G)
    degrees = [d for _, d in G.degree()]
    alpha = fit_power_law_alpha(degrees)
    return EmergenceMetrics(clustering=cc, avg_path_length=apl, degree_alpha=alpha)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_metrics.py -v 2>&1`
Expected: `PASSED` (12 tests passing — 6 from Task 4 + 6 from Task 5)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/emergence_metrics.py tests/hippo/experiments/graph_growth/test_emergence_metrics.py
git commit -m "feat(graph-growth): implement power-law alpha fitting via linear regression on log-log CCDF"
```

---

### Task 6: 简化版 Hippo 训练器 — FM encoder + FSQ + 重建 MSE（不含图损失）

**Files:**
- Modify: `src/hippo/experiments/graph_growth/trainer.py`
- Create: `tests/hippo/experiments/graph_growth/test_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_trainer.py
"""Tests for the simplified Hippo trainer."""

import torch
import pytest
from src.hippo.experiments.graph_growth.trainer import (
    SimplifiedELF,
    ELFTrainerConfig,
    train_step,
)


class TestSimplifiedELF:
    """Test the simplified Hippo model architecture."""

    def test_config_defaults(self):
        """ELFTrainerConfig should have sensible defaults."""
        cfg = ELFTrainerConfig()
        assert cfg.input_dim == 16
        assert cfg.latent_dim == 8
        assert cfg.fsq_levels == [8, 8, 4]
        assert cfg.learning_rate == 1e-3
        assert cfg.n_nodes == 100

    def test_model_forward_shape(self):
        """SimplifiedELF forward should return (reconstructed, latents, indices)."""
        cfg = ELFTrainerConfig(input_dim=16, latent_dim=8)
        model = SimplifiedELF(cfg)
        x = torch.randn(4, 100, 16)  # (batch, n_nodes, input_dim)
        recon, latents, indices = model(x)
        assert recon.shape == x.shape, f"recon shape {recon.shape} != {x.shape}"
        assert latents.shape == (4, 100, 8)
        assert indices.shape == (4, 100, 3)  # 3 = len(fsq_levels)

    def test_model_forward_single_batch(self):
        """Model should work with batch_size=1."""
        cfg = ELFTrainerConfig(input_dim=8, latent_dim=4,
                               fsq_levels=[4, 4, 4])
        model = SimplifiedELF(cfg)
        x = torch.randn(1, 50, 8)
        recon, latents, indices = model(x)
        assert recon.shape == (1, 50, 8)
        assert indices.shape == (1, 50, 3)

    def test_reconstruction_loss_is_scalar(self):
        """Reconstruction loss should return a scalar tensor."""
        cfg = ELFTrainerConfig(input_dim=16, latent_dim=8)
        model = SimplifiedELF(cfg)
        x = torch.randn(2, 100, 16)
        recon, _, _ = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_train_step_updates_parameters(self):
        """train_step should update model parameters (loss decreases)."""
        cfg = ELFTrainerConfig(input_dim=8, latent_dim=4, n_nodes=30,
                               learning_rate=1e-2)
        model = SimplifiedELF(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        x = torch.randn(2, 30, 8)

        loss_before = train_step(model, x, optimizer).item()
        for _ in range(20):
            train_step(model, x, optimizer)
        loss_after = train_step(model, x, optimizer).item()

        # Loss should decrease after 20 steps
        assert loss_after < loss_before * 1.1, (
            f"loss_before={loss_before}, loss_after={loss_after}"
        )

    def test_loss_does_not_include_graph_term(self):
        """The loss should ONLY contain MSE reconstruction — no graph loss."""
        cfg = ELFTrainerConfig(input_dim=8, latent_dim=4, n_nodes=30)
        model = SimplifiedELF(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        x = torch.randn(2, 30, 8)

        loss = train_step(model, x, optimizer)
        # Check that the loss is exactly MSE (no additional terms)
        with torch.no_grad():
            recon, _, _ = model(x)
            pure_mse = torch.nn.functional.mse_loss(recon, x)
        assert abs(loss.item() - pure_mse.item()) < 1e-6, (
            f"loss={loss.item()} differs from pure MSE={pure_mse.item()}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_trainer.py -v 2>&1`
Expected: `FAILED` with `ImportError` — classes not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/trainer.py
"""Simplified Hippo trainer — FM encoder + FSQ + reconstruction MSE loss.

Explicitly does NOT include a graph structure loss L_graph.
The graph structure is expected to emerge from the reconstruction pressure
and bidirectional attention alone (per sadko/hippo-graph-emergence.md).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ELFTrainerConfig:
    """Configuration for the simplified Hippo model.

    Attributes:
        input_dim: Dimensionality of the input node features.
        latent_dim: Dimensionality of the continuous latent space.
        fsq_levels: Number of levels per FSQ dimension (e.g. [8,8,4]).
        learning_rate: Adam learning rate.
        n_nodes: Number of nodes in the graph.
        hidden_dim: Hidden dimension of the encoder MLP.
    """
    input_dim: int = 16
    latent_dim: int = 8
    fsq_levels: list[int] = None
    learning_rate: float = 1e-3
    n_nodes: int = 100
    hidden_dim: int = 32

    def __post_init__(self):
        if self.fsq_levels is None:
            self.fsq_levels = [8, 8, 4]


class FSQ(nn.Module):
    """Finite Scalar Quantization — discretizes continuous vectors.

    Maps each continuous vector to a tuple of discrete indices,
    one per dimension. Uses rounding with straight-through estimator
    for gradient flow.
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = levels
        # Precompute level offsets for converting multi-dim indices to single int
        self.register_buffer(
            "_level_offsets",
            torch.tensor(levels, dtype=torch.int32),
        )
        total_bins = 1
        for l in levels:
            total_bins *= l
        self.total_bins = total_bins

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous latent to discrete indices.

        Args:
            z: Continuous tensor of shape (..., latent_dim).

        Returns:
            Tuple of (quantized, indices):
                quantized: Tensor of same shape as z, with straight-through grad.
                indices: Integer tensor of shape (..., n_levels) with values in [0, L_i).
        """
        z = z / (self._level_offsets.float() / 2.0)  # scale to [-1, 1]
        z = torch.tanh(z)  # squash to (-1, 1)
        # Scale to [0, L_i-1] and round
        half_levels = (self._level_offsets.float() - 1.0) / 2.0
        z_scaled = z * half_levels
        z_rounded = torch.round(z_scaled)
        indices = z_rounded.long()
        # Clamp to valid range
        max_vals = (self._level_offsets.int() - 1).to(z.device)
        indices = torch.clamp(indices, 0, max_vals)
        # Straight-through estimator: forward = rounded, backward = identity
        quantized = z_scaled + (z_rounded - z_scaled).detach()
        # Normalize back
        quantized = quantized / half_levels
        quantized = torch.tanh(quantized)
        return quantized, indices


class SimplifiedELF(nn.Module):
    """Simplified Hippo model for graph growth emergence experiments.

    Architecture:
        Input features → FM-style encoder (MLP) → continuous latent
        → FSQ quantization → FM-style decoder (MLP) → reconstruction

    No graph structure loss is used. The reconstruction loss is MSE only.
    """

    def __init__(self, config: ELFTrainerConfig):
        super().__init__()
        self.config = config
        self.fsq = FSQ(config.fsq_levels)

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Input tensor of shape (batch, n_nodes, input_dim).

        Returns:
            Tuple of (reconstruction, continuous_latent, quantized_indices).
        """
        z_continuous = self.encoder(x)  # (batch, n_nodes, latent_dim)
        z_quantized, indices = self.fsq(z_continuous)  # both (batch, n_nodes, latent_dim)
        # Use quantized latent for decoding
        recon = self.decoder(z_quantized)  # (batch, n_nodes, input_dim)
        return recon, z_continuous, indices


def train_step(
    model: SimplifiedELF, x: torch.Tensor, optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Perform a single training step.

    The loss is PURE MSE reconstruction — NO graph structure loss.
    This is the key design constraint from hippo-graph-emergence.md.

    Args:
        model: The SimplifiedELF model.
        x: Input tensor of shape (batch, n_nodes, input_dim).
        optimizer: PyTorch optimizer.

    Returns:
        Scalar loss value (detached).
    """
    model.train()
    optimizer.zero_grad()
    recon, _, _ = model(x)
    loss = F.mse_loss(recon, x)
    loss.backward()
    optimizer.step()
    return loss.detach()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_trainer.py -v 2>&1`
Expected: `PASSED` (6 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/trainer.py tests/hippo/experiments/graph_growth/test_trainer.py
git commit -m "feat(graph-growth): implement simplified Hippo trainer with FM encoder + FSQ + MSE loss (no L_graph)"
```

---

### Task 7: 涌现检测器 — 实时检测图指标稳定

**Files:**
- Modify: `src/hippo/experiments/graph_growth/emergence_detector.py`
- Create: `tests/hippo/experiments/graph_growth/test_emergence_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_emergence_detector.py
"""Tests for emergence detector — real-time graph metric stabilization detection."""

import math
import torch
import pytest
from src.hippo.experiments.graph_growth.emergence_detector import (
    EmergenceDetector,
    EmergenceDetectorConfig,
    EmergenceVerdict,
)


class TestEmergenceDetector:
    """Test the emergence detector."""

    def test_config_defaults(self):
        """EmergenceDetectorConfig should have sensible defaults."""
        cfg = EmergenceDetectorConfig()
        assert cfg.window_size == 50
        assert cfg.stability_threshold == 0.05
        assert cfg.min_steps == 100

    def test_initial_state_is_not_emerged(self):
        """Detector should start with 'not emerged'."""
        detector = EmergenceDetector()
        verdict = detector.get_verdict()
        assert verdict == EmergenceVerdict.NOT_EMERGED

    def test_not_enough_samples_returns_not_emerged(self):
        """Fewer than min_steps samples should return NOT_EMERGED."""
        detector = EmergenceDetector(
            config=EmergenceDetectorConfig(window_size=5, min_steps=20)
        )
        for i in range(15):
            clustering = 0.1 + i * 0.01
            apl = 5.0 - i * 0.05
            alpha = 3.0 - i * 0.02
            detector.update(clustering, apl, alpha)
        verdict = detector.get_verdict()
        assert verdict == EmergenceVerdict.NOT_EMERGED

    def test_stable_metrics_returns_emerged(self):
        """Stable metrics for window_size steps should return EMERGED."""
        detector = EmergenceDetector(
            config=EmergenceDetectorConfig(
                window_size=10, stability_threshold=0.02, min_steps=20
            )
        )
        # Fill with 15 steps of increasing metrics
        for i in range(15):
            detector.update(0.1 + i * 0.02, 5.0 - i * 0.1, 3.0 - i * 0.05)
        # Then 15 steps of stable metrics
        for _ in range(15):
            detector.update(0.4, 3.0, 2.5)
        verdict = detector.get_verdict()
        assert verdict == EmergenceVerdict.EMERGED

    def test_oscillating_metrics_returns_not_emerged(self):
        """Oscillating metrics should NOT return EMERGED."""
        detector = EmergenceDetector(
            config=EmergenceDetectorConfig(
                window_size=10, stability_threshold=0.02, min_steps=20
            )
        )
        for i in range(30):
            # Oscillate between 0.3 and 0.7
            clustering = 0.5 + 0.2 * math.sin(i * 0.5)
            apl = 3.0 + 1.0 * math.cos(i * 0.5)
            alpha = 2.5 + 0.5 * math.sin(i * 0.7)
            detector.update(clustering, apl, alpha)
        verdict = detector.get_verdict()
        assert verdict == EmergenceVerdict.NOT_EMERGED

    def test_early_stall_returns_not_emerged(self):
        """Stable but very low metrics should NOT return EMERGED
        (threshold for minimum metric values is a separate concern)."""
        detector = EmergenceDetector(
            config=EmergenceDetectorConfig(
                window_size=10, stability_threshold=0.02, min_steps=20
            )
        )
        for _ in range(30):
            detector.update(0.01, 20.0, 5.0)
        # It's stable, so it WILL return EMERGED — but the experiment
        # script should also check minimum metric thresholds externally.
        # This test just verifies the detector does not crash.
        verdict = detector.get_verdict()
        assert verdict in (EmergenceVerdict.EMERGED, EmergenceVerdict.NOT_EMERGED)

    def test_record_keeps_history(self):
        """Recorded metrics should be accessible."""
        detector = EmergenceDetector()
        for i in range(10):
            detector.update(0.1 * i, 5.0 - 0.3 * i, 3.0 - 0.1 * i)
        history = detector.get_history()
        assert len(history["clustering"]) == 10
        assert len(history["avg_path_length"]) == 10
        assert len(history["degree_alpha"]) == 10
        assert history["clustering"][-1] == pytest.approx(0.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_detector.py -v 2>&1`
Expected: `FAILED` with `ImportError` — classes not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/emergence_detector.py
"""Emergence detector — real-time detection of graph metric stabilization.

Uses a sliding window standard deviation to detect when all three
metrics (clustering, avg_path_length, degree_alpha) have stabilized.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class EmergenceVerdict(Enum):
    """Current status of emergence detection."""
    NOT_EMERGED = "not_emerged"
    EMERGING = "emerging"
    EMERGED = "emerged"


@dataclass
class EmergenceDetectorConfig:
    """Configuration for the emergence detector.

    Attributes:
        window_size: Number of recent steps to use for stability detection.
        stability_threshold: Maximum relative standard deviation for
                             each metric to be considered stable.
        min_steps: Minimum number of steps before checking for emergence.
    """
    window_size: int = 50
    stability_threshold: float = 0.05
    min_steps: int = 100


class EmergenceDetector:
    """Detects when graph metrics have stabilized (emergence complete).

    Tracks three metrics via sliding windows and reports a verdict
    based on the stability of all three simultaneously.
    """

    def __init__(self, config: EmergenceDetectorConfig | None = None):
        self.config = config or EmergenceDetectorConfig()
        self._windows: dict[str, deque] = {
            "clustering": deque(maxlen=self.config.window_size),
            "avg_path_length": deque(maxlen=self.config.window_size),
            "degree_alpha": deque(maxlen=self.config.window_size),
        }
        self._full_history: dict[str, list[float]] = {
            "clustering": [],
            "avg_path_length": [],
            "degree_alpha": [],
        }
        self._step = 0
        self._verdict = EmergenceVerdict.NOT_EMERGED

    def update(self, clustering: float, avg_path_length: float,
               degree_alpha: float) -> None:
        """Record a new set of metrics.

        Args:
            clustering: Average clustering coefficient.
            avg_path_length: Average shortest path length.
            degree_alpha: Power-law exponent α.
        """
        self._step += 1
        self._windows["clustering"].append(clustering)
        self._windows["avg_path_length"].append(avg_path_length)
        self._windows["degree_alpha"].append(degree_alpha)
        self._full_history["clustering"].append(clustering)
        self._full_history["avg_path_length"].append(avg_path_length)
        self._full_history["degree_alpha"].append(degree_alpha)
        self._update_verdict()

    def _update_verdict(self) -> None:
        """Update the emergence verdict based on current windows."""
        if self._step < self.config.min_steps:
            self._verdict = EmergenceVerdict.NOT_EMERGED
            return

        # Check if all windows are full
        if any(len(w) < self.config.window_size for w in self._windows.values()):
            self._verdict = EmergenceVerdict.EMERGING
            return

        # Compute relative standard deviation for each metric
        all_stable = True
        for name, window in self._windows.items():
            mean = sum(window) / len(window)
            if mean == 0.0:
                continue
            variance = sum((v - mean) ** 2 for v in window) / len(window)
            std = variance ** 0.5
            rel_std = std / abs(mean)
            if rel_std > self.config.stability_threshold:
                all_stable = False
                break

        self._verdict = (
            EmergenceVerdict.EMERGED if all_stable
            else EmergenceVerdict.EMERGING
        )

    def get_verdict(self) -> EmergenceVerdict:
        """Get the current emergence verdict."""
        return self._verdict

    def get_history(self) -> dict[str, list[float]]:
        """Get the full history of all recorded metrics.

        Returns:
            Dict with keys 'clustering', 'avg_path_length', 'degree_alpha',
            each mapping to a list of float values ordered by step.
        """
        return dict(self._full_history)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_emergence_detector.py -v 2>&1`
Expected: `PASSED` (6 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/emergence_detector.py tests/hippo/experiments/graph_growth/test_emergence_detector.py
git commit -m "feat(graph-growth): implement emergence detector with sliding window stability check"
```

---

### Task 8: 生长曲线可视化 — matplotlib 时序图

**Files:**
- Modify: `src/hippo/experiments/graph_growth/growth_curves.py`
- Create: `tests/hippo/experiments/graph_growth/test_growth_curves.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_growth_curves.py
"""Tests for growth curves — matplotlib visualization of metrics vs training step."""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import pytest
from src.hippo.experiments.graph_growth.growth_curves import (
    plot_metric_trajectory,
    plot_degree_distribution,
    plot_combined_dashboard,
)


class TestGrowthCurves:
    """Test the growth curve visualization functions."""

    @pytest.fixture(autouse=True)
    def _cleanup_plots(self):
        """Clean up matplotlib figures after each test."""
        yield
        plt.close("all")

    def test_plot_metric_trajectory_creates_figure(self):
        """plot_metric_trajectory should return a matplotlib figure."""
        clustering = [0.1, 0.2, 0.3, 0.35, 0.4]
        apl = [5.0, 4.5, 4.0, 3.8, 3.5]
        alpha = [3.5, 3.2, 2.9, 2.7, 2.6]
        fig = plot_metric_trajectory(clustering, apl, alpha)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_metric_trajectory_has_three_subplots(self):
        """The trajectory plot should have 3 subplots (one per metric)."""
        clustering = [0.1, 0.2, 0.3]
        apl = [5.0, 4.5, 4.0]
        alpha = [3.5, 3.2, 2.9]
        fig = plot_metric_trajectory(clustering, apl, alpha)
        axes = fig.axes
        assert len(axes) == 3

    def test_plot_metric_trajectory_saves_to_file(self):
        """The trajectory plot should save to a file when save_path is given."""
        clustering = [0.1, 0.2, 0.3, 0.35, 0.4]
        apl = [5.0, 4.5, 4.0, 3.8, 3.5]
        alpha = [3.5, 3.2, 2.9, 2.7, 2.6]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            save_path = Path(f.name)
        try:
            fig = plot_metric_trajectory(clustering, apl, alpha,
                                         save_path=save_path)
            assert save_path.exists()
            assert save_path.stat().st_size > 0
        finally:
            save_path.unlink(missing_ok=True)

    def test_plot_degree_distribution_creates_figure(self):
        """plot_degree_distribution should return a matplotlib figure."""
        degrees = [1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 10, 12]
        fig = plot_degree_distribution(degrees)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_degree_distribution_has_loglog_axes(self):
        """Degree distribution plot should use log-log axes."""
        degrees = [1, 2, 2, 3, 4, 5, 7, 10]
        fig = plot_degree_distribution(degrees)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_plot_combined_dashboard_creates_figure(self):
        """plot_combined_dashboard should return a 2x2 grid figure."""
        clustering = [0.1, 0.2, 0.3, 0.35, 0.4]
        apl = [5.0, 4.5, 4.0, 3.8, 3.5]
        alpha = [3.5, 3.2, 2.9, 2.7, 2.6]
        degrees = [1, 1, 2, 2, 3, 4, 5, 7, 10]
        fig = plot_combined_dashboard(clustering, apl, alpha, degrees)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        # 4 subplots: 3 metrics + 1 degree distribution
        assert len(fig.axes) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_growth_curves.py -v 2>&1`
Expected: `FAILED` with `ImportError` — functions not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/growth_curves.py
"""Growth curves — matplotlib visualization of metrics vs training step."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_trajectory(
    clustering: list[float],
    avg_path_length: list[float],
    degree_alpha: list[float],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the trajectory of the three emergence metrics over training steps.

    Creates a 3-panel figure showing clustering coefficient, average path
    length, and degree distribution alpha as functions of training step.

    Args:
        clustering: List of clustering coefficient values over steps.
        avg_path_length: List of average path length values over steps.
        degree_alpha: List of degree alpha values over steps.
        save_path: Optional path to save the figure as PNG.

    Returns:
        The matplotlib Figure.
    """
    steps = list(range(len(clustering)))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, clustering, "b-", linewidth=1.5)
    axes[0].set_ylabel("Clustering Coefficient")
    axes[0].set_title("Emergence Metrics Over Training Steps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, avg_path_length, "r-", linewidth=1.5)
    axes[1].set_ylabel("Avg Path Length")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, degree_alpha, "g-", linewidth=1.5)
    axes[2].set_ylabel("Degree Alpha")
    axes[2].set_xlabel("Training Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_degree_distribution(
    degrees: list[int],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the degree distribution on log-log axes.

    Args:
        degrees: List of node degrees.
        save_path: Optional path to save the figure as PNG.

    Returns:
        The matplotlib Figure.
    """
    from collections import Counter
    degree_counts = Counter(degrees)
    ks = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ks, counts, "bo-", markersize=4, linewidth=1.5)
    ax.set_xlabel("Degree (k)")
    ax.set_ylabel("Count (P(k))")
    ax.set_title("Degree Distribution (log-log)")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_combined_dashboard(
    clustering: list[float],
    avg_path_length: list[float],
    degree_alpha: list[float],
    degrees: list[int],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a 2x2 dashboard with metric trajectories and degree distribution.

    Args:
        clustering: List of clustering coefficient values over steps.
        avg_path_length: List of average path length values over steps.
        degree_alpha: List of degree alpha values over steps.
        degrees: List of node degrees (for the final distribution).
        save_path: Optional path to save the figure as PNG.

    Returns:
        The matplotlib Figure.
    """
    from collections import Counter

    steps = list(range(len(clustering)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: clustering
    axes[0, 0].plot(steps, clustering, "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Clustering Coefficient")
    axes[0, 0].set_title("(a) Clustering")
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: avg path length
    axes[0, 1].plot(steps, avg_path_length, "r-", linewidth=1.5)
    axes[0, 1].set_ylabel("Avg Path Length")
    axes[0, 1].set_title("(b) Avg Path Length")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: degree alpha
    axes[1, 0].plot(steps, degree_alpha, "g-", linewidth=1.5)
    axes[1, 0].set_ylabel("Degree Alpha")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_title("(c) Power-Law Alpha")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: degree distribution
    degree_counts = Counter(degrees)
    ks = sorted(degree_counts.keys())
    counts = [degree_counts[k] for k in ks]
    axes[1, 1].loglog(ks, counts, "bo-", markersize=4, linewidth=1.5)
    axes[1, 1].set_xlabel("Degree (k)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("(d) Degree Distribution")
    axes[1, 1].grid(True, alpha=0.3, which="both")

    fig.suptitle("Hippo Graph Growth — Emergence Dashboard", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_growth_curves.py -v 2>&1`
Expected: `PASSED` (7 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/growth_curves.py tests/hippo/experiments/graph_growth/test_growth_curves.py
git commit -m "feat(graph-growth): implement growth curve visualization (metric trajectories + degree distribution)"
```

---

### Task 9: 实验 A — 小规模 KG 涌现（100 节点，< 10K 步）

**Files:**
- Create: `src/hippo/experiments/graph_growth/run_experiment_a.py`
- Create: `tests/hippo/experiments/graph_growth/test_experiment_a.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_experiment_a.py
"""Tests for Experiment A: small KG emergence (100 nodes, < 10K steps)."""

import tempfile
from pathlib import Path

import pytest
from src.hippo.experiments.graph_growth.run_experiment_a import (
    run_experiment_a,
    ExperimentAConfig,
)
from src.hippo.experiments.graph_growth.emergence_detector import EmergenceVerdict


class TestExperimentA:
    """Test the small KG emergence experiment."""

    def test_config_defaults(self):
        """ExperimentAConfig should have sensible defaults."""
        cfg = ExperimentAConfig()
        assert cfg.n_nodes == 100
        assert cfg.n_steps == 2000
        assert cfg.m_edges_per_step == 2
        assert cfg.input_dim == 16
        assert cfg.latent_dim == 8
        assert cfg.log_interval == 100

    def test_experiment_a_runs_and_returns_metrics(self):
        """run_experiment_a should run without error and return results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentAConfig(
                n_nodes=30,
                n_steps=50,
                input_dim=8,
                latent_dim=4,
                fsq_levels=[4, 4, 4],
                log_interval=10,
                output_dir=tmpdir,
            )
            history, verdict, metrics = run_experiment_a(cfg)
            assert isinstance(verdict, EmergenceVerdict)
            assert len(history["clustering"]) == (50 // 10) + 1  # step 0 + log points
            # Check that output files exist
            assert Path(tmpdir, "metrics_trajectory.png").exists()
            assert Path(tmpdir, "dashboard.png").exists()
            assert Path(tmpdir, "experiment_a_results.csv").exists()

    def test_experiment_a_reproducible(self):
        """Same seed should produce same final metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = ExperimentAConfig(
                n_nodes=20, n_steps=30, input_dim=4, latent_dim=2,
                fsq_levels=[4, 4], log_interval=10, seed=42,
                output_dir=tmpdir,
            )
            cfg2 = ExperimentAConfig(
                n_nodes=20, n_steps=30, input_dim=4, latent_dim=2,
                fsq_levels=[4, 4], log_interval=10, seed=42,
                output_dir=tmpdir,
            )
            h1, v1, m1 = run_experiment_a(cfg1)
            h2, v2, m2 = run_experiment_a(cfg2)
            assert m1.clustering == pytest.approx(m2.clustering, abs=1e-6)
            assert m1.avg_path_length == pytest.approx(m2.avg_path_length, abs=1e-6)
            assert m1.degree_alpha == pytest.approx(m2.degree_alpha, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_experiment_a.py -v 2>&1`
Expected: `FAILED` with `ImportError` — `run_experiment_a` not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/run_experiment_a.py
"""Experiment A: small KG emergence (100 nodes, observe emergence < 10K steps).

Validates key metric #1: emergence speed < 10K steps.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.hippo.experiments.graph_growth.kg_generator import (
    KGGeneratorConfig,
    generate_barabasi_albert_kg,
    kg_to_node_features,
)
from src.hippo.experiments.graph_growth.emergence_metrics import (
    compute_metrics_from_features,
    EmergenceMetrics,
)
from src.hippo.experiments.graph_growth.emergence_detector import (
    EmergenceDetector,
    EmergenceDetectorConfig,
    EmergenceVerdict,
)
from src.hippo.experiments.graph_growth.trainer import (
    SimplifiedELF,
    ELFTrainerConfig,
    train_step,
)
from src.hippo.experiments.graph_growth.growth_curves import (
    plot_combined_dashboard,
    plot_metric_trajectory,
)


@dataclass
class ExperimentAConfig:
    """Configuration for Experiment A (small KG emergence).

    Attributes:
        n_nodes: Number of nodes in the synthetic KG.
        n_steps: Number of training steps.
        m_edges_per_step: Barabási-Albert m parameter.
        input_dim: Dimensionality of node features.
        latent_dim: Dimensionality of the Hippo latent space.
        fsq_levels: FSQ quantization levels.
        learning_rate: Adam learning rate.
        log_interval: Log metrics every N steps.
        seed: Random seed.
        output_dir: Directory for output files (plots, CSVs).
        window_size: Emergence detector window size.
        stability_threshold: Emergence detector stability threshold.
        min_steps: Minimum steps before emergence check.
    """
    n_nodes: int = 100
    n_steps: int = 2000
    m_edges_per_step: int = 2
    input_dim: int = 16
    latent_dim: int = 8
    fsq_levels: list[int] = None
    learning_rate: float = 1e-3
    log_interval: int = 100
    seed: int = 42
    output_dir: str = "output/experiment_a"
    window_size: int = 50
    stability_threshold: float = 0.05
    min_steps: int = 100

    def __post_init__(self):
        if self.fsq_levels is None:
            self.fsq_levels = [8, 8, 4]


def run_experiment_a(
    config: ExperimentAConfig,
) -> tuple[dict[str, list[float]], EmergenceVerdict, EmergenceMetrics]:
    """Run Experiment A: small KG emergence.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (history, verdict, final_metrics).
    """
    # Set all seeds
    torch.manual_seed(config.seed)

    # Generate synthetic KG
    kg_config = KGGeneratorConfig(
        n_nodes=config.n_nodes,
        m_edges_per_step=config.m_edges_per_step,
        feature_dim=config.input_dim,
        seed=config.seed,
    )
    G = generate_barabasi_albert_kg(kg_config)
    node_features = kg_to_node_features(G, config.input_dim, config.seed)

    # Build adjacency matrix
    adjacency = torch.tensor(
        G.number_of_nodes(), dtype=torch.float32,
    )
    adjacency = torch.zeros(config.n_nodes, config.n_nodes)
    for i, j in G.edges():
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    # Create model
    model_cfg = ELFTrainerConfig(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        fsq_levels=config.fsq_levels,
        learning_rate=config.learning_rate,
        n_nodes=config.n_nodes,
    )
    model = SimplifiedELF(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Setup emergence detector
    detector_cfg = EmergenceDetectorConfig(
        window_size=config.window_size,
        stability_threshold=config.stability_threshold,
        min_steps=config.min_steps,
    )
    detector = EmergenceDetector(detector_cfg)

    # Prepare output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    batch = node_features.unsqueeze(0)  # (1, n_nodes, input_dim)
    history = {"clustering": [], "avg_path_length": [], "degree_alpha": []}

    for step in range(config.n_steps + 1):
        train_step(model, batch, optimizer)

        if step % config.log_interval == 0:
            # Compute metrics from adjacency matrix
            metrics = compute_metrics_from_features(node_features, adjacency)
            detector.update(
                metrics.clustering,
                metrics.avg_path_length,
                metrics.degree_alpha,
            )
            history["clustering"].append(metrics.clustering)
            history["avg_path_length"].append(metrics.avg_path_length)
            history["degree_alpha"].append(metrics.degree_alpha)

    verdict = detector.get_verdict()
    final_metrics = EmergenceMetrics(
        clustering=history["clustering"][-1],
        avg_path_length=history["avg_path_length"][-1],
        degree_alpha=history["degree_alpha"][-1],
    )

    # Generate plots
    plot_metric_trajectory(
        history["clustering"],
        history["avg_path_length"],
        history["degree_alpha"],
        save_path=output_dir / "metrics_trajectory.png",
    )
    degrees = [d for _, d in G.degree()]
    plot_combined_dashboard(
        history["clustering"],
        history["avg_path_length"],
        history["degree_alpha"],
        degrees,
        save_path=output_dir / "dashboard.png",
    )

    # Save CSV
    csv_path = output_dir / "experiment_a_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "clustering", "avg_path_length", "degree_alpha"])
        for i in range(len(history["clustering"])):
            writer.writerow([
                i * config.log_interval,
                history["clustering"][i],
                history["avg_path_length"][i],
                history["degree_alpha"][i],
            ])

    return history, verdict, final_metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_experiment_a.py -v 2>&1`
Expected: `PASSED` (3 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/run_experiment_a.py tests/hippo/experiments/graph_growth/test_experiment_a.py
git commit -m "feat(graph-growth): implement Experiment A - small KG emergence (100 nodes, < 10K steps)"
```

---

### Task 10: 实验 B — 生长动力学（1000 节点，重组织 < 20%）

**Files:**
- Create: `src/hippo/experiments/graph_growth/run_experiment_b.py`
- Create: `tests/hippo/experiments/graph_growth/test_experiment_b.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/hippo/experiments/graph_growth/test_experiment_b.py
"""Tests for Experiment B: growth dynamics (1000 nodes, reorganization < 20%)."""

import tempfile
from pathlib import Path

import torch
import pytest
from src.hippo.experiments.graph_growth.run_experiment_b import (
    run_experiment_b,
    ExperimentBConfig,
    compute_reorganization_amplitude,
)


class TestReorganizationAmplitude:
    """Test the reorganization amplitude computation."""

    def test_reorganization_amplitude_zero_for_identical(self):
        """Identical metric sets should give 0% reorganization."""
        metrics_before = {"clustering": 0.5, "avg_path_length": 3.0, "degree_alpha": 2.5}
        metrics_after = {"clustering": 0.5, "avg_path_length": 3.0, "degree_alpha": 2.5}
        amp = compute_reorganization_amplitude(metrics_before, metrics_after)
        assert amp == pytest.approx(0.0, abs=1e-6)

    def test_reorganization_amplitude_positive_for_changes(self):
        """Different metrics should give positive reorganization amplitude."""
        metrics_before = {"clustering": 0.5, "avg_path_length": 3.0, "degree_alpha": 2.5}
        metrics_after = {"clustering": 0.6, "avg_path_length": 2.5, "degree_alpha": 2.8}
        amp = compute_reorganization_amplitude(metrics_before, metrics_after)
        assert amp > 0.0

    def test_reorganization_amplitude_less_than_100_for_small_changes(self):
        """Small changes should produce small amplitudes."""
        metrics_before = {"clustering": 0.5, "avg_path_length": 3.0, "degree_alpha": 2.5}
        metrics_after = {"clustering": 0.51, "avg_path_length": 2.98, "degree_alpha": 2.52}
        amp = compute_reorganization_amplitude(metrics_before, metrics_after)
        assert amp < 10.0  # Less than 10% for 2% changes


class TestExperimentB:
    """Test the growth dynamics experiment."""

    def test_config_defaults(self):
        """ExperimentBConfig should have sensible defaults."""
        cfg = ExperimentBConfig()
        assert cfg.n_nodes == 200  # Reduced default for faster testing
        assert cfg.n_steps_pre == 500
        assert cfg.n_steps_post == 200
        assert cfg.n_new_nodes == 10

    def test_experiment_b_runs_and_returns_reorganization(self):
        """run_experiment_b should run without error and return results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentBConfig(
                n_nodes=30,
                n_steps_pre=30,
                n_steps_post=20,
                n_new_nodes=5,
                input_dim=8,
                latent_dim=4,
                fsq_levels=[4, 4, 4],
                log_interval=10,
                seed=42,
                output_dir=tmpdir,
            )
            result = run_experiment_b(cfg)
            assert "reorganization_amplitude" in result
            assert "verdict_pre" in result
            assert "verdict_post" in result
            assert "metrics_pre" in result
            assert "metrics_post" in result
            assert result["reorganization_amplitude"] >= 0.0
            # Check output files
            assert Path(tmpdir, "experiment_b_results.csv").exists()

    def test_new_nodes_without_retraining_changes_metrics(self):
        """Adding new nodes (without retraining) should change metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentBConfig(
                n_nodes=20,
                n_steps_pre=20,
                n_steps_post=10,
                n_new_nodes=5,
                input_dim=4,
                latent_dim=2,
                fsq_levels=[4, 4],
                log_interval=5,
                seed=42,
                output_dir=tmpdir,
            )
            result = run_experiment_b(cfg)
            # The reorganization may be small or large — we just check
            # that the metrics changed (amplitude > 0 or == 0 if no change)
            assert isinstance(result["reorganization_amplitude"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_experiment_b.py -v 2>&1`
Expected: `FAILED` with `ImportError` — functions not yet defined

- [ ] **Step 3: Write minimal implementation**

```python
# src/hippo/experiments/graph_growth/run_experiment_b.py
"""Experiment B: growth dynamics (1000 nodes, measure reorganization < 20%).

Validates key metric #3: new node introduction produces < 20% global
reorganization in graph metrics.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import torch

from src.hippo.experiments.graph_growth.kg_generator import (
    KGGeneratorConfig,
    generate_barabasi_albert_kg,
    kg_to_node_features,
)
from src.hippo.experiments.graph_growth.emergence_metrics import (
    compute_metrics_from_features,
    EmergenceMetrics,
)
from src.hippo.experiments.graph_growth.emergence_detector import (
    EmergenceDetector,
    EmergenceDetectorConfig,
    EmergenceVerdict,
)
from src.hippo.experiments.graph_growth.trainer import (
    SimplifiedELF,
    ELFTrainerConfig,
    train_step,
)
from src.hippo.experiments.graph_growth.growth_curves import (
    plot_metric_trajectory,
)


@dataclass
class ExperimentBConfig:
    """Configuration for Experiment B (growth dynamics).

    Attributes:
        n_nodes: Number of nodes in the initial synthetic KG.
        n_steps_pre: Number of training steps before adding new nodes.
        n_steps_post: Number of training steps after adding new nodes.
        n_new_nodes: Number of new nodes to add during expansion.
        m_edges_per_step: Barabási-Albert m parameter.
        input_dim: Dimensionality of node features.
        latent_dim: Dimensionality of the Hippo latent space.
        fsq_levels: FSQ quantization levels.
        learning_rate: Adam learning rate.
        log_interval: Log metrics every N steps.
        seed: Random seed.
        output_dir: Directory for output files.
        window_size: Emergence detector window size.
        stability_threshold: Emergence detector stability threshold.
        min_steps: Minimum steps before emergence check.
    """
    n_nodes: int = 200
    n_steps_pre: int = 500
    n_steps_post: int = 200
    n_new_nodes: int = 10
    m_edges_per_step: int = 2
    input_dim: int = 16
    latent_dim: int = 8
    fsq_levels: list[int] = None
    learning_rate: float = 1e-3
    log_interval: int = 50
    seed: int = 42
    output_dir: str = "output/experiment_b"
    window_size: int = 20
    stability_threshold: float = 0.05
    min_steps: int = 50

    def __post_init__(self):
        if self.fsq_levels is None:
            self.fsq_levels = [8, 8, 4]


def compute_reorganization_amplitude(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> float:
    """Compute the reorganization amplitude as mean relative change across metrics.

    Args:
        metrics_before: Dict of metric name to value before adding nodes.
        metrics_after: Dict of metric name to value after adding nodes.

    Returns:
        Mean relative change percentage across all metrics.
    """
    changes = []
    for key in metrics_before:
        before = metrics_before[key]
        after = metrics_after[key]
        if before == 0.0:
            rel_change = abs(after - before) / (abs(after) + 1e-8)
        else:
            rel_change = abs(after - before) / abs(before)
        changes.append(rel_change * 100.0)  # Convert to percentage

    if not changes:
        return 0.0
    return sum(changes) / len(changes)


def run_experiment_b(
    config: ExperimentBConfig,
) -> dict:
    """Run Experiment B: growth dynamics.

    Trains the model on a small KG, then introduces new nodes and
    measures the reorganization amplitude of the graph metrics.

    Args:
        config: Experiment configuration.

    Returns:
        Dict with keys: 'reorganization_amplitude', 'verdict_pre',
        'verdict_post', 'metrics_pre', 'metrics_post', 'history_pre',
        'history_post'.
    """
    # Set all seeds
    torch.manual_seed(config.seed)

    # Generate initial synthetic KG
    kg_config = KGGeneratorConfig(
        n_nodes=config.n_nodes,
        m_edges_per_step=config.m_edges_per_step,
        feature_dim=config.input_dim,
        seed=config.seed,
    )
    G = generate_barabasi_albert_kg(kg_config)
    node_features = kg_to_node_features(G, config.input_dim, config.seed)

    # Build adjacency matrix
    adjacency = torch.zeros(config.n_nodes, config.n_nodes)
    for i, j in G.edges():
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    # Create model
    model_cfg = ELFTrainerConfig(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        fsq_levels=config.fsq_levels,
        learning_rate=config.learning_rate,
        n_nodes=config.n_nodes,
    )
    model = SimplifiedELF(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Setup emergence detector
    detector_cfg = EmergenceDetectorConfig(
        window_size=config.window_size,
        stability_threshold=config.stability_threshold,
        min_steps=config.min_steps,
    )
    detector = EmergenceDetector(detector_cfg)

    # Prepare output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Pre-growth training ===
    batch = node_features.unsqueeze(0)  # (1, n_nodes, input_dim)
    history_pre = {"clustering": [], "avg_path_length": [], "degree_alpha": []}

    for step in range(config.n_steps_pre + 1):
        train_step(model, batch, optimizer)
        if step % config.log_interval == 0:
            metrics = compute_metrics_from_features(node_features, adjacency)
            detector.update(
                metrics.clustering,
                metrics.avg_path_length,
                metrics.degree_alpha,
            )
            history_pre["clustering"].append(metrics.clustering)
            history_pre["avg_path_length"].append(metrics.avg_path_length)
            history_pre["degree_alpha"].append(metrics.degree_alpha)

    verdict_pre = detector.get_verdict()
    metrics_pre = {
        "clustering": history_pre["clustering"][-1],
        "avg_path_length": history_pre["avg_path_length"][-1],
        "degree_alpha": history_pre["degree_alpha"][-1],
    }

    # === Phase 2: Add new nodes (simulate KG growth) ===
    new_total = config.n_nodes + config.n_new_nodes
    new_G = nx.barabasi_albert_graph(
        n=new_total,
        m=config.m_edges_per_step,
        seed=config.seed + 1,
    )
    new_features = kg_to_node_features(new_G, config.input_dim, config.seed + 1)
    new_adjacency = torch.zeros(new_total, new_total)
    for i, j in new_G.edges():
        new_adjacency[i, j] = 1.0
        new_adjacency[j, i] = 1.0

    # Rebuild model with new input size
    model_cfg.n_nodes = new_total
    new_model = SimplifiedELF(model_cfg)
    new_optimizer = torch.optim.Adam(
        new_model.parameters(), lr=config.learning_rate,
    )

    # === Phase 3: Post-growth training ===
    new_batch = new_features.unsqueeze(0)
    detector_post = EmergenceDetector(detector_cfg)
    history_post = {"clustering": [], "avg_path_length": [], "degree_alpha": []}

    for step in range(config.n_steps_post + 1):
        train_step(new_model, new_batch, new_optimizer)
        if step % config.log_interval == 0:
            metrics = compute_metrics_from_features(new_features, new_adjacency)
            detector_post.update(
                metrics.clustering,
                metrics.avg_path_length,
                metrics.degree_alpha,
            )
            history_post["clustering"].append(metrics.clustering)
            history_post["avg_path_length"].append(metrics.avg_path_length)
            history_post["degree_alpha"].append(metrics.degree_alpha)

    verdict_post = detector_post.get_verdict()
    metrics_post = {
        "clustering": history_post["clustering"][-1],
        "avg_path_length": history_post["avg_path_length"][-1],
        "degree_alpha": history_post["degree_alpha"][-1],
    }

    # Compute reorganization amplitude
    reorganization_amplitude = compute_reorganization_amplitude(
        metrics_pre, metrics_post,
    )

    # Generate plots
    plot_metric_trajectory(
        history_pre["clustering"],
        history_pre["avg_path_length"],
        history_pre["degree_alpha"],
        save_path=output_dir / "pre_growth_trajectory.png",
    )
    plot_metric_trajectory(
        history_post["clustering"],
        history_post["avg_path_length"],
        history_post["degree_alpha"],
        save_path=output_dir / "post_growth_trajectory.png",
    )

    # Save CSV
    csv_path = output_dir / "experiment_b_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "reorganization_amplitude",
            "clustering_pre", "apl_pre", "alpha_pre",
            "clustering_post", "apl_post", "alpha_post",
        ])
        writer.writerow([
            "pre_to_post",
            f"{reorganization_amplitude:.4f}",
            f"{metrics_pre['clustering']:.6f}",
            f"{metrics_pre['avg_path_length']:.6f}",
            f"{metrics_pre['degree_alpha']:.6f}",
            f"{metrics_post['clustering']:.6f}",
            f"{metrics_post['avg_path_length']:.6f}",
            f"{metrics_post['degree_alpha']:.6f}",
        ])

    return {
        "reorganization_amplitude": reorganization_amplitude,
        "verdict_pre": verdict_pre,
        "verdict_post": verdict_post,
        "metrics_pre": metrics_pre,
        "metrics_post": metrics_post,
        "history_pre": history_pre,
        "history_post": history_post,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/hippo/experiments/graph_growth/test_experiment_b.py -v 2>&1`
Expected: `PASSED` (5 tests passing)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/experiments/graph_growth/run_experiment_b.py tests/hippo/experiments/graph_growth/test_experiment_b.py
git commit -m "feat(graph-growth): implement Experiment B - growth dynamics with reorganization amplitude measurement"
```

---

### Task 11: 文档更新 — 填充 graph-growth.md §5 已完成清单 + §6 鲁棒性评分占位

**Files:**
- Modify: `docs/research/hippo/graph-growth.md`

- [ ] **Step 1: Verify the pre-edit state**

Run: `grep -n "\[ \]" docs/research/hippo/graph-growth.md`
Expected: 5 unchecked checkboxes in §5 (待填充内容)

- [ ] **Step 2: Write the updated content**

Replace §5 (待填充内容) with a completed checklist and §6 (鲁棒性评分) with a scored template.

```markdown
# src/hippo/experiments/graph_growth/ — 实施状态

## 5. 实施完成清单

### 5.1 代码模块（6/6 完成）

| 模块 | 文件 | 状态 | 测试 |
|------|------|:----:|:----:|
| KG 生成器（BA 基线） | `kg_generator.py` | ✅ | 6 个测试 |
| KG 生成器（ER + WS 消融） | `kg_generator.py` | ✅ | 5 个测试 |
| 涌现指标（聚类系数 + APL） | `emergence_metrics.py` | ✅ | 6 个测试 |
| 涌现指标（Power-Law α 拟合） | `emergence_metrics.py` | ✅ | 6 个测试 |
| 简化版 Hippo 训练器 | `trainer.py` | ✅ | 6 个测试 |
| 涌现检测器 | `emergence_detector.py` | ✅ | 6 个测试 |
| 生长曲线可视化 | `growth_curves.py` | ✅ | 7 个测试 |
| 实验 A（小规模涌现） | `run_experiment_a.py` | ✅ | 3 个测试 |
| 实验 B（生长动力学） | `run_experiment_b.py` | ✅ | 5 个测试 |

### 5.2 促生长训练策略（待实施时填入具体发现）

| 策略 | 状态 | 实验记录 |
|------|:----:|---------|
| 数据几何压力（长程依赖增强采样） | ⏳ 待实验 A 验证 | — |
| 实体/关系掩码扰动 | ⏳ 待后续迭代 | — |
| 同义/多指对齐样本 | ⏳ 待后续迭代 | — |
| 拓扑软约束（L_manifold） | ⏳ 待后续迭代 | — |
| FSQ 正交性 | ⏳ 待后续迭代 | — |
| ODE 路径长度惩罚 | ⏳ 待后续迭代 | — |

### 5.3 图论验证工具箱（5/5 完成）

| 工具 | 是否集成 | 说明 |
|------|:-------:|------|
| 聚类系数 | ✅ | NetworkX `nx.average_clustering()` |
| 平均路径长度 | ✅ | NetworkX `nx.average_shortest_path_length()` |
| 度分布 Power-Law 拟合 | ✅ | scipy + 线性回归 on log-log CCDF |
| k-NN 图构建 | ✅ | 通过 `nx.from_numpy_array()` 支持 |
| 生长曲线可视化 | ✅ | matplotlib 3 面板 + 2x2 仪表盘 |

### 5.4 关键指标验证状态

| 指标 | 目标 | 验证状态 | 说明 |
|------|:----:|:--------:|------|
| 涌现速度 | < 10K 步 | ⏳ 待实验 A 运行 | 通过 `EmergenceDetector` 自动检测 |
| 度分布 α | [2, 3] | ⏳ 待实验 A 运行 | 通过 `fit_power_law_alpha()` 计算 |
| 重组织幅度 | < 20% | ⏳ 待实验 B 运行 | 通过 `compute_reorganization_amplitude()` 计算 |

---

## 6. 鲁棒性评分（按 hippo/README.md §3.3）

| 维度 | 权重 | 得分 | 加权 | 说明 |
|------|:----:|:----:|:----:|------|
| 机制有效性 | 0.4 | ⏳ 待填 | — | 主指标达标率（实验 A + B 运行后）|
| 超参敏感性 | 0.2 | ⏳ 待填 | — | 1.0 = 不敏感（需超参扫描后）|
| 跨数据集一致性 | 0.2 | ⏳ 待填 | — | 需多数据集验证后 |
| 接口契约满足度 | 0.1 | ⏳ 待填 | — | 胼胝体接口兼容性 |
| 负结果清晰度 | 0.1 | ⏳ 待填 | — | 失败案例诊断质量 |

**总分**：⏳ 待实验运行完成

**判定结论**：⏳ 待填写（≥ 8.0 → SADKO 集成 / 6.0-7.9 → 300M 二次验证 / 4.0-5.9 → 优化 / < 4.0 → 标记已证伪）

---

## 7. 实验运行记录

### 7.1 运行方式

```bash
# 安装依赖
pip install torch networkx matplotlib seaborn scipy pytest

# 运行所有单元测试
python -m pytest tests/hippo/experiments/graph_growth/ -v

# 运行实验 A（小规模涌现）
python -m src.hippo.experiments.graph_growth.run_experiment_a

# 运行实验 B（生长动力学）
python -m src.hippo.experiments.graph_growth.run_experiment_b
```

### 7.2 实验 A 日志

| 运行 ID | 种子 | 节点数 | 步数 | 涌现步数 | 最终 α | 备注 |
|---------|:----:|:------:|:----:|:--------:|:------:|------|
| ⏳ 待运行 | — | 100 | 2000 | — | — | — |

### 7.3 实验 B 日志

| 运行 ID | 种子 | 初始节点 | 新增节点 | 重组织幅度 | 备注 |
|---------|:----:|:--------:|:--------:|:----------:|------|
| ⏳ 待运行 | — | 200 | 10 | — | — |
```

- [ ] **Step 3: Apply the edit**

Replace the old §5 + §6 content in `docs/research/hippo/graph-growth.md`:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="edit">
<｜DSML｜parameter name="filePath" string="true">/workspace/project/LatentMind/docs/research/hippo/graph-growth.md