from enum import Enum

class MetricType(Enum):
    MAHALANOBIS = "Mahalanobis"
    EMBEDDING = "Embedding"
    IOU = "IoU"

class Metric:
    def __init__(self, metric_type: MetricType):
        self.metric_type = metric_type
        self.metric = self._build_metric(metric_type)

    def _build_metric(self, metric_type: MetricType):
        if metric_type == MetricType.MAHALANOBIS:
            return  MahalanobisMetric()
        elif metric_type == MetricType.EMBEDDING:
            return EmbeddingMetric()
        elif metric_type == MetricType.IOU:
            return IoUMetric()
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
