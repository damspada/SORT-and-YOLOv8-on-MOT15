# PATH CONFIGURATION
# SEQ = "Venice-1"
# SEQ = "KITTI-17"
# SEQ = "ADL-Rundle-6"
SEQ = "PETS09-S2L1"
# SEQ = "TUD-Stadtmitte"

# SEQ = "KITTI-13"
# SEQ = "ADL-Rundle-8"
# SEQ = "ETH-Bahnhof"
# SEQ = "ETH-Sunnyday"
# SEQ = "ETH-Pedcross2"
# SEQ = "TUD-Campus"

INPUT_LOCAL = f"/home/spada/Documenti/Uni/MOT15/{SEQ}"
OUTPUT_LOCAL = f"/home/spada/Documenti/Uni/SORT-output/{SEQ}/"

INPUT_COLAB = f"/content/MOT15/test/{SEQ}"
OUTPUT_COLAB = f"/content/SORT_output/{SEQ}/"

# DATASET AND MODEL CONFIGURATION
SEQ_INFO = "seqinfo.ini"           # Sequence information file name
YOLO_VERSION = "yolov8m.pt"        # YOLOv8 model variant (nano for speed)

# TRACKING PARAMETERS
# Track Lifecycle Management
MAX_FRAME_LOST = 10                 # Max consecutive frames before track deletion
MIN_HITS = 5                       # Minimum hits before track is considered confirmed
# Detection and Matching Thresholds
THRESHOLD_IOU = 0.75               # IoU threshold for Hungarian matching (0-1)
MIN_CONF = 0.35                    # Minimum YOLO detection confidence (0-1)
# Statistical Matching Parameters
ALPHA_CHI = 0.95                   # Chi-squared confidence level for Mahalanobis (0-1)

# KALMAN FILTER CONFIGURATION
# Process Noise Covariance Matrix (Q) - Motion Model Uncertainty
Q_POSITION_NOISE = 4               # Position noise variance (pixels²)
Q_DIMENSION_NOISE = 50             # Dimension noise variance (pixels²)
Q_VELOCITY_NOISE = 100             # Velocity noise variance (pixels/frame)²
# Initial State Covariance Matrix (P) - Initial Estimate Uncertainty
P_POSITION_INITIAL = 9             # Initial position uncertainty (pixels²)
P_DIMENSION_INITIAL = 4            # Initial dimension uncertainty (pixels²)
P_VELOCITY_INITIAL = 900           # Initial velocity uncertainty (pixels/frame)²