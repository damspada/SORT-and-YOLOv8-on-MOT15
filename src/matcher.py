import torch
import networkx as nx
from typing import Dict, Tuple, List, Set
from networkx.algorithms.bipartite import hopcroft_karp_matching
from networkx.algorithms.bipartite import to_vertex_cover

class Matcher:
  """
  The matching is done with the Hungarian algorithm and the IoU metric
  """
  def __init__(self):
    pass
  
  # def _IoU(A: torch.Tensor, B: torch.Tensor) -> float:
  #   """
  #   Shape of A and B is [x_sx, y_sx, x_dx, y_dx]
  #   """
  #   intersection_x = max(0, min(A[2], B[2] - max(A[0], B[0])))
  #   intersection_y = max(0, min(A[3], B[3] - max(A[1], B[1])))
  #   intersection_boxes = intersection_x * intersection_y
  #   if intersection_boxes:
  #     wA, wB = A[2]-A[0], B[2]-B[0]
  #     hA, hB = A[3]-A[1], B[3]-B[1]
  #     union_boxes = (wA * hA) + (wB * hB) - intersection_boxes
  #     return intersection_boxes/union_boxes
  #   else:
  #     return intersection_boxes
  
  @staticmethod
  def _IoU_matrix(tracks: torch.Tensor, detections: torch.Tensor) -> torch.Tensor:
    """
    tracks shape:     (N,4)
    detections shape: (M,4)
    output shape:     (N,M)
    """
    # Unsqueeze enables broadcasting (requires 2+ dimensions)
    tracks = tracks.unsqueeze(1)         #(N,1,4)
    detections = detections.unsqueeze(0) #(1,M,4)

    # C[i:j] = max(tracks[i], detections[j])
    x_sx = torch.maximum(tracks[:,:,0], detections[:,:,0]) #(N,M)
    y_sx = torch.maximum(tracks[:,:,1], detections[:,:,1]) #(N,M)
    x_dx = torch.minimum(tracks[:,:,2], detections[:,:,2]) #(N,M)
    y_dx = torch.minimum(tracks[:,:,3], detections[:,:,3]) #(N,M)

    # Calculate intersection and set negative value to zero (no intersection)
    intersection_x = torch.clamp(x_dx - x_sx, min=0)     #(N,M)
    intersection_y = torch.clamp(y_dx - y_sx, min=0)     #(N,M)
    intersection_boxes = intersection_x * intersection_y #(N,M)

    # Calculate the union area for each track and detection
    tracks_areas = (tracks[:,:,2] - tracks[:,:,0]) * (tracks[:,:,3] - tracks[:,:,1]) #(N,1)
    detection_areas = (detections[:,:,2] - detections[:,:,0]) * (detections[:,:,3] - detections[:,:,1]) #(1,M)
    union_boxes = tracks_areas + detection_areas - intersection_boxes #(N,M)

    # The Hungarian algorithm minimizes cost, so we use 1 - IoU
    return torch.ones_like(union_boxes) - (intersection_boxes / union_boxes) #(N,M)

  @staticmethod
  def _rectangle_to_square(matrix: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,M) in:
      (N,N) if N > M
      (M,M) if M > N
    All new entries are set to 1e6 to discourage assignments by the algorithm
    """
    N, M = matrix.shape[0], matrix.shape[1]
    if N > M:
      matrix = torch.cat([matrix, torch.full((N, N-M), 1e6)], dim=1)
    if M > N:
      matrix = torch.cat([matrix, torch.full((M-N, M), 1e6)], dim=0)
    return matrix

  @staticmethod
  def _subtract_on_dimensions(H: torch.Tensor, dim: int):
    mins, _ = H.min(dim = dim, keepdim=True)
    return H - mins

  @staticmethod
  def _matrix_to_graph(H: torch.Tensor) -> Tuple[nx.Graph, Set]:
    G = nx.Graph()
    row = set()
    for (i,j) in torch.nonzero(H == 0, as_tuple=False):
      G.add_edge(f"R{i}", f"C{j}")
      row.add(f"R{i}")
    return G, row

  @staticmethod
  def _analyze_vertex_cover(dim:int, vertex_cover: Set[str]) -> Dict[str, torch.Tensor]:
    row_uncovered = list(range(dim))
    row_covered = []
    column_uncovered = list(range(dim))
    column_covered = []
    for elem in vertex_cover:
      index = int(elem[1])
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
    return [(int(r[1]),int(c[1])) for r, c in matching.items() if r.startswith("R")]

  def _update_matrix_with_min_uncovered(self, H: torch.Tensor, vertex_cover: Set[str]) -> torch.Tensor:
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

  def _maximum_matching_minimum_vertex_cover(self, H: torch.Tensor) -> Tuple[Dict[str,str], Set[str]]:
    G, T = self._matrix_to_graph(H)
    matching = hopcroft_karp_matching(G, top_nodes=T)
    vertex_cover = to_vertex_cover(G, matching, top_nodes=T)
    return self._unique_matching(matching), vertex_cover

  def _match_tracks_and_detections():
    return

  def hungarian_algorithm(self, tracks: torch.Tensor, detections: torch.Tensor):

    # Step 0 -> Build hungarian matrix (N,N) with the cost
    H = self._rectangle_to_square(self._IoU_matrix(tracks, detections))

    # Step 1 -> Subtract from each row the minimum element in it
    H_current = self._subtract_on_dimensions(H, dim=1)

    # Step 2 -> Subtract from each column the minimum element in it
    H_current = self._subtract_on_dimensions(H_current, dim=0)

    # Step 3 -> Cross the 0's with the minimum number of lines needed (if N==#lines jump to 5)
    matching, vertex_cover = self._maximum_matching_minimum_vertex_cover(H_current)

    # Step 4 -> Find the smallest entry not covered by any line,
    #           subtract this entry to the not covered row  and
    #           add this entry to the covered collumn (jump to 3)
    while len(vertex_cover) < H.shape[0]:
      H_current = self._update_matrix_with_min_uncovered(H_current, vertex_cover)
      matching, vertex_cover = self._maximum_matching_minimum_vertex_cover(H_current)    
    
    # Step 5 -> Assign detections to tracks starting with the line with only one zero,
    #           do not accept pairs with a cost greater than a threshold.
    

    return H
    






