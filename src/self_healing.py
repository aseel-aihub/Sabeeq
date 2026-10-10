# ============================================================================
#  Sabeeq
#  Author : Aseel  (GitHub: aseel-aihub)
#  Copyright (c) 2026 Aseel. All rights reserved.
# ============================================================================
"""
self_healing.py — the power network repairs itself (Objective 5).

The network is a graph (nodes + lines). When a node is destroyed, we use
Dijkstra to find the cheapest new path from any working power plant
to every facility.

Result for each facility:
    - unaffected : power path did not change
    - rerouted   : got power through a new path (self-healed)
    - lost       : no power path left
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from src.facilities import load_facilities

__all__ = [
    "NetworkStatus",
    "SUPPLY_SOURCES",
    "build_network",
    "supply_sources",
    "simulate_failure",
    "node_positions",
]

# Power plants = where electricity starts.
SUPPLY_SOURCES = ["Power_Plant_A", "Power_Plant_B"]

# Lines of the network. weight = cost (distance / loss).
_EDGES: list[tuple[str, str, float]] = [
    ("Power_Plant_A", "Substation_North", 10),
    ("Power_Plant_A", "Substation_West", 14),
    ("Power_Plant_B", "Substation_South", 12),
    ("Power_Plant_B", "Substation_East", 16),
    # Ring lines between substations — these make self-healing possible.
    ("Substation_North", "Substation_East", 9),
    ("Substation_East", "Substation_South", 8),
    ("Substation_South", "Substation_West", 11),
    ("Substation_West", "Substation_North", 7),
    # Link between the two power plants.
    ("Power_Plant_A", "Power_Plant_B", 20),
]


@dataclass
class NetworkStatus:
    """Network state after the damage."""

    damaged_nodes: list
    supplied: dict = field(default_factory=dict)      # facility -> path
    unsupplied: list = field(default_factory=list)    # facilities without power
    rerouted: dict = field(default_factory=dict)      # facility -> (old, new)
    baseline_cost: dict = field(default_factory=dict) # cost before damage
    current_cost: dict = field(default_factory=dict)  # cost after damage

    @property
    def healed_count(self) -> int:
        """How many facilities were saved by rerouting."""
        return len(self.rerouted)

    @property
    def total_facilities(self) -> int:
        return len(self.supplied) + len(self.unsupplied)

    @property
    def availability(self) -> float:
        """Percentage of facilities that still have power."""
        if self.total_facilities == 0:
            return 0.0
        return len(self.supplied) / self.total_facilities


def build_network() -> nx.Graph:
    """
    Build the graph and connect each facility to its power source.

    We use an undirected graph because real power lines work both ways.
    (My first version used a directed graph, and that blocked some
    valid reroutes.)
    """
    G = nx.Graph()

    for u, v, w in _EDGES:
        G.add_edge(u, v, weight=w, kind="transmission")

    for _, fac in load_facilities().iterrows():
        G.add_node(fac["facility_id"], kind="facility", name=fac["name"],
                   criticality=int(fac["criticality"]))
        G.add_edge(fac["facility_id"], fac["supply_node"], weight=3, kind="feeder")

        # Backup line has a higher cost, so it's only used when the main
        # path is gone — that is the self-healing behaviour.
        backup = fac.get("backup_node")
        if isinstance(backup, str) and backup.strip():
            G.add_edge(fac["facility_id"], backup.strip(), weight=9, kind="backup_feeder")

    return G


def supply_sources(G: nx.Graph) -> list[str]:
    """Power plants that still exist in the graph."""
    return [n for n in SUPPLY_SOURCES if n in G]


def _cheapest_supply(G: nx.Graph, facility_id: str):
    """
    Cheapest path to the facility from any working power plant.

    Returns (path, cost), or (None, None) if no path exists.
    """
    best_path, best_cost = None, None

    for source in supply_sources(G):
        try:
            cost, path = nx.single_source_dijkstra(G, source, target=facility_id, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue  # no path from this plant

        if best_cost is None or cost < best_cost:
            best_path, best_cost = path, cost

    return best_path, best_cost


def simulate_failure(damaged_nodes) -> NetworkStatus:
    """
    Remove the damaged nodes and calculate power again for every facility.
    """
    # Accept one name or a list of names.
    if isinstance(damaged_nodes, str):
        damaged_nodes = [damaged_nodes]
    damaged_nodes = [str(n) for n in damaged_nodes]

    baseline = build_network()   # before damage
    damaged = build_network()    # after damage

    # Unknown names are ignored safely.
    damaged.remove_nodes_from([n for n in damaged_nodes if n in damaged])

    status = NetworkStatus(damaged_nodes=damaged_nodes)

    facility_ids = [n for n, d in baseline.nodes(data=True) if d.get("kind") == "facility"]
    for facility_id in facility_ids:
        base_path, base_cost = _cheapest_supply(baseline, facility_id)
        status.baseline_cost[facility_id] = base_cost

        # If the facility itself was hit, it has no power.
        if facility_id in damaged_nodes:
            status.unsupplied.append(facility_id)
            continue

        new_path, new_cost = _cheapest_supply(damaged, facility_id)

        if new_path is None:
            status.unsupplied.append(facility_id)
            continue

        status.supplied[facility_id] = new_path
        status.current_cost[facility_id] = new_cost

        # Different path than before = it was rerouted (self-healed).
        if base_path is not None and new_path != base_path:
            status.rerouted[facility_id] = (base_path, new_path)

    return status


def node_positions(G: nx.Graph, seed: int = 7) -> dict:
    """
    Fixed drawing positions so the graph doesn't jump around every refresh.
    """
    return nx.spring_layout(G, seed=seed, k=0.55, iterations=200)
