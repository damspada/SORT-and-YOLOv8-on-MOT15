import torch
from typing import List

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
    tracks_areas = (tracks[:,:,2] - tracks[:,:,0]) * (tracks[:,:,2] - tracks[:,:,0]) #(N,1)
    detection_areas = (detections[:,:,2] - detections[:,:,0]) * (detections[:,:,2] - detections[:,:,0]) #(1,M)
    union_boxes = tracks_areas * detection_areas - intersection_boxes #(N,M)

    # The Hungarian algorithm minimizes cost, so we use 1 - IoU
    return torch.ones_like(union_boxes) - (intersection_boxes / union_boxes) #(N,M)

  def _rectangle_to_square(matrix: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,M) in:
      (N,N) if N > M
      (M,M) if M > N
    All new entries are set to 1e6 to discourage assignments by the algorithm
    """

  def match_tracks_and_detections(self, tracks: torch.Tensor, detections: torch.Tensor):
    # Step 0 -> Build hungarian matrix (N,N) with the cost
    H = self._rectangle_to_square(self._IoU_matrix(tracks, detections))
    # Step 1 -> Subtract from each row the minimum element in it
    # Step 2 -> Subtract from each column the minimum element in it
    # Step 3 -> Cross the 0's with the minimum number of lines needed (if N==#lines jump to 5)
    # Step 4 -> Find the smallest entry not covered by any line
    #           and subtract this entry to the entire matrix except 0 (jump to 3)
    # Step 5 -> Assign detections to tracks starting with the line with only one zero,
    #           do not accept pairs with a cost greater than a threshold.
    






