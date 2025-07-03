# The MOSTLY AI Prize: Sequential Data Challenge Submission

This repository contains the source code for a submission to "The MOSTLY AI Prize" sequential data challenge.

This solution consists of a multi-stage pipeline that first generates a large, diverse pool of synthetic sequential data and then uses a post-processing strategy to select an optimal subset of sequences with high statistical accuracy and temporal coherence.

## Methodology

The core of this solution is a two-phase process designed to capture the multi-column and sequential characteristics of the original data.

### 1\. Synthetic Data Pool Generation

To create a diverse foundation for the final dataset, the pipeline first generates a large pool of synthetic sequences (groups). This is achieved by training a generative model from MOSTLY AI.

Instead of relying on a single training run, the pipeline iteratively trains the model multiple times. Each training cycle produces a new, distinct generator that contributes its own unique synthetic sequences to the overall data pool.

### 2\. Post-processing for Group Selection

Once the data pool is generated, a multi-step selection algorithm carefully chooses the final 20,000 groups. This is where the "magic" happens, refining the raw generated data into a high-quality subset by optimizing for both sequential integrity and statistical fidelity.

1.  **Coherence-Based Pre-selection**: The process begins with a refinement step that focuses exclusively on the *coherence* of the generated sequences. An iterative swapping algorithm selects an initial subset of groups that best matches the original data's sequential characteristics, specifically the distributions of "categories per sequence" and "sequences per category". This ensures the fundamental sequential structure of the data is preserved.
2.  **Iterative Statistical Refinement**: The coherence-optimized subset is processed by a final, intensive refinement for accuracy. The algorithm iteratively swaps entire groups from the subset with "better" groups from the larger data pool to minimize the univariate, bivariate, and trivariate statistical errors (measured by L1 distance).

## Key Features

This pipeline was designed to be robust, efficient, and adaptable, making it a strong foundation for various synthetic data generation tasks.

  * **High Generalizability**: The two-stage architecture (generation and post-processing) is modular. The post-processing scripts can be easily reused for other sequential datasets with minimal changes. The parameters exposed in `main.py` (e.g., `top_k_pairs`, `coherence_iterations`) can tune the trade-off between compute time and quality.

  * **Model Agnostic**: The refinement pipeline works with **any source of synthetic sequential data**. While this project uses the `mostlyai` SDK for the initial pool generation, you could substitute it with any generative model. The strength of the final output comes from the post-processing, greatly enhancing the quality of any base synthetic dataset.

  * **Performance-Optimized**: The entire post-processing pipeline is engineered for efficiency:

      * **Low Memory Footprint**: By converting data into binned, integer-based representations and extensively using `scipy.sparse` matrices to manage per-group contributions, the algorithms operate with very low memory overhead. The core logic is further accelerated with `numba` for time-critical computations.
      * **CPU-Friendly Refinement**: The entire multi-stage refinement process is CPU-bound.

  * **Privacy Guarantees**: By selecting from a vast, pre-generated pool based on aggregate statistical distributions (up to the trivariate level), the risk of replicating individual records or their sensitive combinations is minimal. The final dataset copies the *statistical patterns* of the original data, not the data points themselves, ensuring privacy-safe output.

## System Requirements

  - **Python**: 3.10+
  - **GPU Environment**: This submission should run with a GPU in an **`g5.2xlarge`** instance.
  - **AWS AMI**: The `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)` was used for testing this solution.
  - **Disk Space**: To be on the safe side, a minimum of **75 GB** of disk space is recommended.
  - **NVIDIA Driver**: Before running, ensure the NVIDIA driver is correctly installed by executing `nvidia-smi`.

## Setup & Usage

The entire pipeline is orchestrated by the `run.sh` script, which handles environment setup, dependency installation, and execution.

### 1\. Grant Execute Permissions

First, make the `run.sh` script executable:

```bash
chmod +x run.sh
```

### 2\. Run the Pipeline

Execute the script, providing the full path to the training data CSV file as an argument.

```bash
./run.sh /path/to/your/sequential-training.csv
```

The script will:

1.  Check for `uv` and install it if not present.
2.  Create a local virtual environment in a `.venv` directory.
3.  Install all required Python packages from `requirements.txt`.
4.  Activate the virtual environment.
5.  Run the main pipeline script (`main.py`) with the provided data path.

## Output

The pipeline generates two data outputs:

  * **Intermediate Data Pool**: A large CSV file containing all generated sequences is saved in the `pool_data/` directory.
  * **Final Submission File**: The final, refined synthetic dataset as a CSV is saved in the `results/` directory with a timestamped filename, e.g., `seq_result_20250702_1955.csv`. This is the file that should be used for evaluation. For additional information, an HTML report of the generated data and the training data is generated in the same folder.

## License

This project is licensed under the **MIT License**.