import torch
from scipy.stats import chi2
from enum import Enum
from typing import List, Tuple
from ultralytics.engine.results import Boxes

from src.predictor import Predictor
from src.track import Track
from variables import THRESHOLD_IOU, ALPHA_CHI
from src.matrixUtils import MatrixUtils


class MetricType(Enum):
  IOU = "IoU"
  EMBEDDING = "Embedding"
  MAHALANOBIS = "Mahalanobis"


class Metric:
  def __init__(self, metric_type: MetricType):
    self.metric_type = metric_type
    self.metric = self._build_metric(metric_type)

  def _build_metric(self, metric_type: MetricType):
    if metric_type == MetricType.MAHALANOBIS:
      return  Mahalanobis_metric()
    elif metric_type == MetricType.EMBEDDING:
      return Embedding_metric()
    elif metric_type == MetricType.IOU:
      return IoU_metric()


class IoU_metric:
  def __call__(self, tracks: List[Track], detections: Boxes) -> torch.Tensor:
    """
    Computes the pairwise IoU cost between tracks and detections.
    Input tensors must contain bounding boxes in [x1, y1, x2, y2] format.
    """
    tracks_xyxy = MatrixUtils.tracks_to_matrix_xyxy(tracks)
    return self.iou_matrix(tracks_xyxy, detections.xyxy)

  @staticmethod
  def iou_matrix(tracks: torch.Tensor, detections: torch.Tensor) -> Tuple(torch.Tensor, int):
    """
    tracks shape:     (N,4)
    detections shape: (M,4)
    output shape:     (N,M)
    Create a matrix where rows represent tracks, columns represent detections, 
    and each element is a cost estimating how likely a track is associated with a detection.
    Lower values indicate a better match.
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
    H = torch.ones_like(union_boxes) - (intersection_boxes / union_boxes) #(N,M)
    return H, THRESHOLD_IOU


class Mahalanobis_metric:
  def __call__(self, tracks: List[Track], detections: Boxes) -> torch.Tensor:
    tracks_xywh, P = MatrixUtils.tracks_to_matrix_xywh_and_P(tracks)
    return self.mahalabobis_matrix(tracks_xywh, detections.xywh, P)

  @staticmethod
  def mahalabobis_matrix(HX: torch.Tensor, D: torch.Tensor, P: torch.Tensor) -> Tuple(torch.Tensor, int):
    diff = D.unsqueeze(0) - HX.unsqueeze(1)  # (1, M, 4) - (N, 1, 4) = (N, M, 4)

    HP = Predictor.H.unsqueeze(0) @ P         # (N,4,6) @ (6,6) = (N,4,6)
    HP_HT = HP @ Predictor.H.transpose(1,2)   # (N,4,6) @ (6,4) = (N,4,4)
    S =  HP_HT + Predictor.R                  # (N,4,4) + (4,4) = (N,4,4)
    S_inv = torch.linalg.inv(S)               # (N, 4, 4)

    # Rivedere bene questa funzione
    mahal = torch.einsum('nij,njk,nij->ni', diff, S_inv, diff)  # (N, M)

    k = HX.shape[1]
    alpha = ALPHA_CHI
    threshold = chi2.ppf(alpha, df=k)

    return mahal, threshold
    

class Embedding_metric:
  def __call__(self, tracks: List[Track], detections: Boxes) -> torch.Tensor:
    pass



