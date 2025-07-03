import torch

from typing import Dict, Tuple, List, Set
from ultralytics.engine.results import Boxes

import networkx as nx
from networkx.algorithms.bipartite import hopcroft_karp_matching
from networkx.algorithms.bipartite import to_vertex_cover

from src.track import Track
from src.metrics import Metric
from src.metrics import MetricType
from src.matrixUtils import MatrixUtils

class Matcher:
  """
  The matching is done with the Hungarian algorithm and a metric.
  """

  @staticmethod
  def _subtract_on_dimensions(H: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Subtracts the minimum value along the specified dimension from each element in that dimension.
    """
    mins, _ = H.min(dim = dim, keepdim=True)
    return H - mins


  @staticmethod
  def _matrix_to_graph(H: torch.Tensor) -> Tuple[nx.Graph, Set]:
    """
    Converts a matrix into a bipartite graph where each row represents a left node, 
    each column represents a right node, and edges exist only for matrix elements equal to zero.
    """
    G = nx.Graph()
    row = set()
    for (i,j) in torch.nonzero(H == 0, as_tuple=False):
      G.add_edge(f"R{i}", f"C{j}")
      row.add(f"R{i}")
    return G, row


  @staticmethod
  def _analyze_vertex_cover(dim_matrix:int, vertex_cover: Set) -> Dict:
    """
    Given a vertex cover represented as a list, constructs a dictionary that explicitly 
    stores the covered and uncovered rows and columns for easy access and further processing.
    """
    row_uncovered = list(range(dim_matrix))
    row_covered = []
    column_uncovered = list(range(dim_matrix))
    column_covered = []
    for elem in vertex_cover:
      index = int(elem[1:])
      if elem[0] == "R":
        row_uncovered.remove(index)
        row_covered.append(index)
      elif elem[0] == "C":
        column_uncovered.remove(index)
        column_covered.append(index)
    return {
      "row_uncovered": torch.tensor(row_uncovered),
      "row_covered" : torch.tensor(row_covered),
      "column_uncovered" : torch.tensor(column_uncovered),
      "column_covered" : torch.tensor(column_covered)
      }


  @staticmethod
  def _unique_matching(matching: Dict[str,str]) -> List[Tuple[str,str]]:
    """
    Tansforms a dictionary representing a maximum matching into a list of unique pairs, 
    conrtaining all matches from tracks to detections without duplicates.
    """
    return [(int(r[1:]),int(c[1:])) for r, c in matching.items() if r.startswith("R")]


  def _update_matrix_with_min_uncovered(self, H: torch.Tensor, vertex_cover: Set[str]) -> torch.Tensor:
    """
    Subtracts the minimum uncovered value from all uncovered rows and adds it to all covered columns.
    Resets any negative values in the matrix to zero.
    """
    # Calculate the minimum uncovered entry
    vc_dict = self._analyze_vertex_cover(H.shape[0], vertex_cover)
    sub_H = H.index_select(0, vc_dict["row_uncovered"]).index_select(1, vc_dict["column_uncovered"])
    min_entry = sub_H.min()
    # Update the Hungarian Matrix
    H[vc_dict["row_uncovered"], :] -= min_entry
    H[:, vc_dict["column_covered"]] += min_entry
    # Trasform all negative number to zero
    H = H = torch.clamp(H, min=0)
    return H


  def _maximum_matching_minimum_vertex_cover(self, H: torch.Tensor) -> Tuple[Dict, Set]:
    """
    Computes the maximum matching and the minimum vertex cover of a bipartite graph.
    """
    G, T = self._matrix_to_graph(H)
    matching = hopcroft_karp_matching(G, top_nodes=T)
    vertex_cover = to_vertex_cover(G, matching, top_nodes=T)
    matching = self._unique_matching(matching)
    return matching, vertex_cover


  @staticmethod #Rimuovere???
  def _assign_detections_with_threshold(H: torch.Tensor, matching: List, original_dim: Tuple, threshold:int):
    """
    Processes the matching to produce track-detection pairs, lost tracks in the current frame,
    and new detections. Also discards assignments involving auxiliary dimensions. 
    """
    results = {
      "assignments" : [],
      "lost_tracks" : [],
      "new_detections" : []
    }
    for (r, c) in matching:
      if H[r,c] <= threshold:
        results["assignments"].append((r, c))
      else:
        if r < original_dim[0]:
          results["lost_tracks"].append(r)
        if c < original_dim[1]:
          results["new_detections"].append(c)
    return results


  def hungarian_algorithm(self, tracks: List[Track], detections: Boxes, metric_type: MetricType) -> Dict:
    """
    Implements the Hungarian algorithm in 5 steps to solve the assignment problem.
    Tracks and detections must be in the form [x1,y1,x2,y2]
    """
    # Step 0 -> Build hungarian matrix (N,N) with the cost
    metric = Metric(metric_type)
    H, threshold = MatrixUtils.rectangle_to_square(metric.metric(tracks, detections))

    # Step 1 -> Subtract from each row the minimum element in it
    H_current = self._subtract_on_dimensions(H, dim=1)

    # Step 2 -> Subtract from each column the minimum element in it
    H_current = self._subtract_on_dimensions(H_current, dim=0)

    # Step 3 -> Cross the 0's with the minimum number of lines needed (if N==#lines jump to 5)
    matching, vertex_cover = self._maximum_matching_minimum_vertex_cover(H_current)

    # Step 4 -> Subtract the smallest uncovered entry to uncovered row and add it to covered column
    while len(vertex_cover) < H.shape[0]:
      H_current = self._update_matrix_with_min_uncovered(H_current, vertex_cover)
      matching, vertex_cover = self._maximum_matching_minimum_vertex_cover(H_current)    
    
    # Step 5 -> Assign detections to tracks, don't accept the pairs with high cost
    #RIVEDERE
    original_dim = (len(tracks), detections.shape[0])
    results = self._assign_detections_with_threshold(H, matching, original_dim, threshold)

    return results
    






