from src.SORTTrackers import SORTTrackers

from variables import INPUT_COLAB
from variables import OUTPUT_COLAB

from variables import INPUT_LOCAL
from variables import OUTPUT_LOCAL

if __name__ == "__main__":
  sort = SORTTrackers()
  sort.run_tracking(
    input_path = INPUT_COLAB,
    output_path = OUTPUT_COLAB
  )
