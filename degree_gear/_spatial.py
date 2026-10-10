"""Spatial neighbourhood helpers used by Degree-GEAR.

The functions below come from the AAAI-27 code supplement
(``degree_gear/tools/spatial.py`` and ``degree_gear/tools/context_embedding.py``).
Since 0.2.0 the per-spot hex builder is replaced by the linear-time
``_build_all_hex_layers`` (same layers, same order), and ``_get_top_genes``
raises ``ValueError`` instead of exiting the interpreter.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _get_hex_neighbors(row: int, col: int, sparse: bool = True) -> List[Tuple[int, int]]:
    """
    Get hexagonal grid neighbors for a given spot.
    
    Parameters
    ----------
    row : int
        Array row coordinate
    col : int
        Array column coordinate
    sparse : bool
        If True, use Visium sparse hex pattern (spots at every other position).
        If False, use dense hex pattern (adjacent spots).
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (row, col) tuples for neighboring spots
    """
    if sparse:
        # Visium sparse hexagonal pattern: spots are at every other position
        # In Visium, spots are spaced by 2 in columns and rows alternate offset
        # This creates the physical hexagonal arrangement
        if row % 2 == 0:  # Even row
            offsets = [
                (-1, -1),  # Upper-Left
                (-1, 1),   # Upper-Right  
                (0, -2),   # Left
                (0, 2),    # Right
                (1, -1),   # Lower-Left
                (1, 1),    # Lower-Right
            ]
        else:  # Odd row
            offsets = [
                (-1, -1),  # Upper-Left
                (-1, 1),   # Upper-Right
                (0, -2),   # Left
                (0, 2),    # Right
                (1, -1),   # Lower-Left
                (1, 1),    # Lower-Right
            ]
    else:
        # Dense hexagonal pattern (original implementation)
        if row % 2 == 0:  # Even row
            offsets = [
                (-1, 0),   # North-East
                (0, 1),    # East
                (1, 0),    # South-East
                (1, -1),   # South-West
                (0, -1),   # West
                (-1, -1)   # North-West
            ]
        else:  # Odd row
            offsets = [
                (-1, 1),   # North-East
                (0, 1),    # East
                (1, 1),    # South-East
                (1, 0),    # South-West
                (0, -1),   # West
                (-1, 0)    # North-West
            ]
    
    neighbors = []
    for d_row, d_col in offsets:
        neighbor_row = row + d_row
        neighbor_col = col + d_col
        neighbors.append((neighbor_row, neighbor_col))
    
    return neighbors


def _get_hex_neighbors_with_distance(
    row: int, 
    col: int, 
    max_distance: int = 1,
    sparse: bool = True
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Get hexagonal neighbors at multiple distances.
    
    In a hexagonal grid, neighbors follow a specific pattern:
    - Layer 1: 6 neighbors (immediate neighbors)
    - Layer 2: 12 neighbors (second ring)
    - Layer 3: 18 neighbors (third ring)
    - Pattern: Each layer n has 6*n neighbors
    
    Note: For Visium sparse grids, actual neighbor counts may be lower
    due to the spaced arrangement of spots.
    
    Parameters
    ----------
    row : int
        Center spot row coordinate
    col : int
        Center spot column coordinate
    max_distance : int
        Maximum neighbor distance (1 = immediate neighbors only)
    sparse : bool
        If True, use Visium sparse hex pattern
        
    Returns
    -------
    Dict[int, List[Tuple[int, int]]]
        Dictionary with distance as key and list of coordinates as value
    """
    neighbors_by_distance = {}
    visited = {(row, col)}
    current_layer = [(row, col)]
    
    for distance in range(1, max_distance + 1):
        next_layer = []
        distance_neighbors = []
        
        for curr_row, curr_col in current_layer:
            for neighbor in _get_hex_neighbors(curr_row, curr_col, sparse=sparse):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_layer.append(neighbor)
                    distance_neighbors.append(neighbor)
        
        neighbors_by_distance[distance] = distance_neighbors
        current_layer = next_layer
        
        # Log the actual vs expected neighbor count for debugging
        expected_count = 6 * distance
        actual_count = len(distance_neighbors)
        if actual_count != expected_count:
            logger.debug(
                f"Layer {distance}: Found {actual_count} neighbors "
                f"(expected {expected_count} for ideal hex grid, "
                f"sparse={sparse})"
            )
    
    return neighbors_by_distance


def _extract_visium_coordinates(adata: ad.AnnData) -> Optional[pd.DataFrame]:
    """
    Extract Visium array coordinates from AnnData object.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with array_row and array_col columns, or None if not found
    """
    if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
        coords = pd.DataFrame({
            'array_row': adata.obs['array_row'].values,
            'array_col': adata.obs['array_col'].values
        }, index=adata.obs_names)
        return coords
    return None


def _build_all_hex_layers(
    adata: ad.AnnData,
    n_layers: int = 3
) -> Dict[int, Dict[int, List[int]]]:
    """
    Build hexagonal grid layers for every observation of Visium data.

    Returns a dictionary mapping each cell index to {layer number: cell indices}.
    The grid-position lookup is built once, so the cost is linear in the number
    of spots. A grid position shared by several observations resolves to the
    first of them, and observation names must be unique (checked upstream).
    """
    visium_coords = _extract_visium_coordinates(adata)
    if visium_coords is None:
        return {cell_idx: {} for cell_idx in range(adata.n_obs)}

    rows = visium_coords['array_row'].values
    cols = visium_coords['array_col'].values
    position_of = {}
    for pos, spot in enumerate(zip(rows, cols)):
        position_of.setdefault(spot, pos)

    all_layers = {}
    for cell_idx in range(len(visium_coords)):
        center_row = int(visium_coords.iloc[cell_idx]['array_row'])
        center_col = int(visium_coords.iloc[cell_idx]['array_col'])
        neighbors_by_dist = _get_hex_neighbors_with_distance(center_row, center_col, n_layers)
        layers = {}
        for distance, coords_list in neighbors_by_dist.items():
            layer_indices = [position_of[spot] for spot in coords_list if spot in position_of]
            if layer_indices:
                layers[distance] = layer_indices
        all_layers[cell_idx] = layers
    return all_layers


def _build_delaunay_adjacency(
    spatial_coords: np.ndarray,
) -> sp.csr_matrix:
    from scipy.spatial import Delaunay
    from scipy.sparse import lil_matrix

    tri = Delaunay(spatial_coords)
    adjacency = lil_matrix((len(spatial_coords), len(spatial_coords)))
    for simplex in tri.simplices:
        for left in range(3):
            for right in range(left + 1, 3):
                i, j = simplex[left], simplex[right]
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    return adjacency.tocsr()


def _build_topological_layers_from_adjacency(
    adjacency: sp.csr_matrix,
    cell_idx: int,
    n_layers: int = 3
) -> Dict[int, List[int]]:
    layers = {}
    visited = {cell_idx}
    current_layer = [cell_idx]
    for layer_num in range(1, n_layers + 1):
        next_layer = []
        for cell in current_layer:
            for neighbor in adjacency[cell].nonzero()[1]:
                if neighbor not in visited:
                    next_layer.append(neighbor)
                    visited.add(neighbor)
        if next_layer:
            layers[layer_num] = list(set(next_layer))
        current_layer = next_layer
    return layers


def _build_all_topological_layers(
    adjacency: sp.csr_matrix,
    n_layers: int = 3,
) -> Dict[int, Dict[int, List[int]]]:
    return {
        cell_idx: _build_topological_layers_from_adjacency(
            adjacency, cell_idx, n_layers
        )
        for cell_idx in range(adjacency.shape[0])
    }


def _get_top_genes(
    expression: np.ndarray,
    gene_names: List[str],
    top_n: int = 10,
    method: Literal['top_n', 'threshold'] = 'top_n',
    threshold: float = 0.0
) -> List[str]:
    """
    Get top expressed genes from expression array.
    
    Returns list of gene names sorted by expression (highest first).
    """
    if method == 'top_n':
        # Get top N genes by expression
        top_indices = np.argsort(expression)[-top_n:][::-1]
        return [gene_names[i] for i in top_indices if expression[i] > 0]
    elif method == 'threshold' and threshold is not None:
        # Get all genes above threshold, sorted by expression
        above_threshold = np.where(expression > threshold)[0]
        sorted_indices = above_threshold[np.argsort(expression[above_threshold])[::-1]]
        return [gene_names[i] for i in sorted_indices[:top_n]]
    else:
        raise ValueError("method must be 'top_n', or 'threshold' with a threshold value")
