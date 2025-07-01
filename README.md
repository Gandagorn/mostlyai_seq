# The MOSTLY AI Prize: Sequential Data Challenge Submission

This repository contains an open-source solution for the Sequential Data Challenge of The MOSTLY AI Prize. The pipeline generates synthetic sequential data by training a generative model and then applying a multi-stage post-processing and refinement algorithm to optimize for statistical accuracy and coherence.

## Requirements

All required Python packages are listed in the `requirements.txt` file. The `run.sh` script will automatically create a virtual environment and install them using `uv`.

  - Python 3.12.3
  - `uv` (will be installed automatically by the run script if not present)

## Usage

The entire pipeline can be executed using the provided shell script. It handles environment setup, dependency installation, and running the main Python script.

1.  **Make the script executable:**

    ```bash
    chmod +x run.sh
    ```

2.  **Run the script:**
    Provide the full path to the training data CSV file as the first argument.

    ```bash
    ./run.sh /path/to/your/training-data.csv
    ```

    For example:

    ```bash
    ./run.sh ./data/sequential-training.csv
    ```

The final synthetic dataset will be saved as `seq_result_{timestamp}.csv` in the `results/` directory.

## Hardware Specification.
  * `c5d.12xlarge`