import sys
from src.SORTTrackers import SORTTrackers
from src.metrics import MetricType

from variables import INPUT_COLAB
from variables import OUTPUT_COLAB

from variables import INPUT_LOCAL
from variables import OUTPUT_LOCAL

def main():
  print(f"Starting tracking...")

  try:
    sort = SORTTrackers()
    sort.run_tracking(
        input_path=INPUT_LOCAL,
        metric=MetricType.IOU,
        output_path=OUTPUT_LOCAL
    )
    print("Tracking completed successfully!")
    
  except KeyboardInterrupt:
      print("\nTracking interrupted by user.")
  except Exception as e:
      print(f"Error during tracking: {e}")
      sys.exit(1)

if __name__ == "__main__":
    main()