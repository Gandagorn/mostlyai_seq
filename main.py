from datetime import datetime
import os
import random
import pandas as pd
import numpy as np
import argparse

from mostlyai import qa

from pipeline.training import generate_data
from pipeline.postprocessing import run_refinement
from pipeline.utils import coherence_report_columns, calculate_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the sequential data generation pipeline.")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the input training CSV file.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with a smaller dataset and fewer iterations.'
    )
    args = parser.parse_args()

    # set seeds
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    start_time = datetime.now()

    timestamp = start_time.strftime("%Y%m%d_%H%M")
    pool_dir = "pool_data"
    results_dir = "results"
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    pool_file_name = os.path.join(pool_dir, f"seq_intermediate_pre_trained_pool_{timestamp}.csv")
    result_output = os.path.join(results_dir, f"seq_result_{timestamp}.csv")
    report_output = os.path.join(results_dir, f"report_{timestamp}.html")

    print(f"Starting sequential pipeline with timestamp: {timestamp}")

    # load in data
    train_df = pd.read_csv(args.data_path)
    group_size = train_df["group_id"].nunique()

    model_params = dict(
        training_oversampling_size=4,
        model_name='MOSTLY_AI/Medium',
        batch_size=None,
        gradient_accumulation_steps=5,
        max_sequence_window=16,
        enable_flexible_generation=False,
        value_protection=True,
        rare_category_replacement_method="SAMPLE",
        max_epochs=16,
        max_training_time=60*3, # 3 hours
        sample_size=300_000,
        train_iterations=3,
    )

    postprocessing_params = dict(
        swap_size=100,
        top_k_pairs=45,
        top_k_triples=140,
        coherence_iterations=10_000,
        swap_size_multiplier=8,
        swap_iterations=1_200,
        coherence_max_time=90,  # 1 1/2 hour
        refinement_max_time=60,  # 1 hour
    )

    is_test = args.test
    if is_test:
        print("Testing mode enabled: Using smaller dataset and fewer iterations.")
        train_df = train_df[train_df["group_id"].isin(train_df["group_id"].drop_duplicates().sample(5_000))]
        model_params["sample_size"] = 50_000
        model_params["max_training_time"] = 1
        postprocessing_params["swap_size"] = 100
        postprocessing_params["top_k_pairs"] = 10
        postprocessing_params["top_k_triples"] = 5
        postprocessing_params["coherence_iterations"] = 1_000
        postprocessing_params["coherence_max_time"] = 2
        postprocessing_params["refinement_max_time"] = 2

    print(f"Training with {len(train_df)} samples and {train_df['group_id'].nunique()} groups.")

    print("--- STEP 1: Generating Synthetic Data Pool ---")
    synthetic_data = generate_data(
        train_df=train_df,
        model_params=model_params
    )
    synthetic_data.to_csv(pool_file_name, index=False)
    print(f"Synthetic data pool saved to {pool_file_name}")

    print("--- STEP 2: Post-processing and Refinement ---")
    synthetic_pool = pd.read_csv(pool_file_name)
    subset_df = run_refinement(
        synthetic_pool=synthetic_pool,
        train_df=train_df,
        params=postprocessing_params,
    )

    print("--- STEP 3: Final Evaluation and Sanity Checks ---")
    print(f"Number of samples: {len(subset_df)}")
    print(f"Number of groups: {subset_df['group_id'].nunique()}")

    sequence_lengths = {}
    for df, name in [(train_df, "train"), (subset_df, "subset_df")]:
        group_dist = df["group_id"].value_counts().value_counts().sort_index()
        sequence_lengths[name] = group_dist / group_dist.sum()

    print(f"Sequence lengths comparison:\n{pd.DataFrame(sequence_lengths)}")

    accuracy = calculate_accuracy(
        original_data=train_df.drop(columns="group_id"),
        synthetic_data=subset_df.drop(columns="group_id"),
    )
    coherence = coherence_report_columns(train_df, subset_df)

    print("--- Final Evaluation Scores ---")
    print("  Using Local Validation Metrics")
    combined_score = 0.25 * coherence + 0.75 * accuracy.get('overall_accuracy', 0)
    print(f"  Coherence      : {coherence:.4f}")
    print(f"  Accuracy       : {accuracy.get('overall_accuracy', 0):.4f}")
    print("  --------------------")
    for key, value in accuracy.items():
        if key != 'overall_accuracy':
            readable_key = key.replace('_', ' ').title()
            print(f"    - {readable_key:<20}: {value:.4f}")
    print(f"  Combined Score : {combined_score:.4f}")
    print("---------------------------------")

    report, metrics = qa.report(
        trn_tgt_data=train_df.copy(deep=True),
        syn_tgt_data=subset_df.copy(deep=True),
        tgt_context_key="group_id",
        report_path=report_output,
    )
    print(f"Mostly QA Report accuracy {metrics.accuracy.overall}")
    print(f"QA report saved to {report_output}")

    print("--- STEP 4: Storing Final Result ---")
    print(f"Storing result to {result_output}")
    subset_df.to_csv(result_output, index=False)

    duration_hours = (datetime.now() - start_time).total_seconds() / (60 * 60)
    print(f"Pipeline completed successfully in {duration_hours:.2f} hours.")

