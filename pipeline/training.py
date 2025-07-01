import pandas as pd
import logging
from mostlyai.sdk import MostlyAI
from .utils import calculate_accuracy, coherence_report_columns

logger = logging.getLogger(__name__)

def train_generator(
        train_df,
        model_params,
        mostly,
):
    """Trains a Mostly AI generator for sequential data.

    This function prepares data for sequential model training with oversampled groups for performance.
    It then configures and runs the training process with optimized parameters.

    Args:
        train_df: The input DataFrame containing sequential data with a group_id.
        model_params: A dictionary of model and training configurations.
        mostly: An initialized MostlyAI SDK client instance.

    Returns:
        A trained Mostly AI generator object for sequential data.
    """
    dataframes_to_concat = []     # oversample and do not elongate groups
    for i in range(model_params["training_oversampling_size"]):
        temp_df = train_df.copy(deep=True)
        temp_df["group_id"] = temp_df["group_id"] + f"_{i}"
        dataframes_to_concat.append(temp_df)

    oversampled = pd.concat(dataframes_to_concat, ignore_index=True)
    logger.info(f"Training with {len(oversampled)} samples and {oversampled['group_id'].nunique()} groups")
    oversampled_groups = oversampled[['group_id']].drop_duplicates()

    g = mostly.train(config={
        'tables': [{
            'name': 'groups',
            'data': oversampled_groups,
            'primary_key': 'group_id',
        }, {
            'name': 'events',
            'data': oversampled,
            'foreign_keys': [{
                'column': 'group_id',
                'referenced_table': 'groups',
                'is_context': True,
            }],
            'tabular_model_configuration': {
                'model': model_params["model_name"],
                'batch_size': model_params["batch_size"],
                'gradient_accumulation_steps': model_params["gradient_accumulation_steps"],
                'max_sequence_window': model_params["max_sequence_window"],
                'enable_flexible_generation': model_params["enable_flexible_generation"],
                'value_protection': model_params["value_protection"],
                "rare_category_replacement_method": model_params["rare_category_replacement_method"],
                'max_epochs': model_params["max_epochs"],
                'max_training_time': model_params["max_training_time"],
            }
        }]
    })
    return g

def generate_data(
    train_df: pd.DataFrame,
    model_params: dict,
):
    """Generates a pool of synthetic sequential data.

    This function orchestrates the sequential data generation process. It can
    run multiple training and generation iterations. In each iteration, it calls
    `train_generator` to create a new model and then samples from it to generate
    synthetic sequences. The results are concatenated to create a large pool.

    Args:
        train_df: The original training DataFrame with sequential data.
        model_params: A dictionary of parameters for training and generation,
                      including the number of iterations and sample size.

    Returns:
        A DataFrame containing the combined pool of generated synthetic data.
    """
    mostly = MostlyAI(local=True)
    iterations = model_params["train_iterations"]
    synthetic_data_list = []

    for i in range(iterations):
        g = train_generator(train_df, model_params, mostly)

        sd = mostly.generate(
            g,
            config={
                'tables': [{
                    'name': 'groups',
                    'configuration': {
                        "sample_size": model_params["sample_size"]//iterations,
                    }
                }, {
                    'name': 'events',
                    'configuration': {
                        'enable_data_report': True,
                    }
                }]
            }
        )

        synthetic_data_dict = sd.data()
        synthetic_data_it = synthetic_data_dict['events']
        synthetic_data_list.append(synthetic_data_it)
        acc = calculate_accuracy(
            original_data=train_df.drop(columns="group_id"),
            synthetic_data=synthetic_data_it.drop(columns="group_id"),
        )
        coherence = coherence_report_columns(train_df, synthetic_data_it)
        logger.info(f"Iteration {i+1}: Generated data accuracy {acc.get('overall_accuracy', 0):.6f} and coherence {coherence:.6f}")

    synthetic_data = pd.concat(synthetic_data_list)
    acc = calculate_accuracy(
        original_data=train_df.drop(columns="group_id"),
        synthetic_data=synthetic_data.drop(columns="group_id"),
    )
    coherence = coherence_report_columns(train_df, synthetic_data)
    logger.info(f"Final combined data pool: Accuracy {acc.get('overall_accuracy', 0):.6f} and coherence {coherence:.6f}")
    return synthetic_data