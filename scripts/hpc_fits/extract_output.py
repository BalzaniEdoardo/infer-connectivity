import json
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


def collect_fit_results(output_dir: pathlib.Path) -> pd.DataFrame:
    """
    Collect all fit results from the output directory and create a comprehensive DataFrame.

    Args:
        output_dir: Path to directory containing cv_results, metadata, and model files

    Returns:
        DataFrame with columns for scores, regularization, model info, and file paths
    """

    results = []
    output_dir = pathlib.Path(output_dir)

    # Find all CV results files
    cv_files = list(output_dir.glob("cv_results_*.npz"))
    print(f"Found {len(cv_files)} CV results files")

    for cv_file in cv_files:
        # Parse filename to extract identifiers
        filename_parts = cv_file.stem.split('_')
        dataset_name = filename_parts[2]  # dataset_<name>
        neuron_id = filename_parts[4]  # neuron_<id>
        config_name = '_'.join(filename_parts[6:])  # config_<name>

        # Load CV results
        cv_data = np.load(cv_file)

        # Find corresponding metadata file
        metadata_file = output_dir / f"metadata_{dataset_name}_neuron_{neuron_id}_config_{config_name}.json"
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found for {cv_file.name}")
            print(f"metadata file name: {metadata_file.name}")
            print(metadata_file in list(output_dir.glob("metadata_*.json")))
            continue

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        print("Loaded metadata file.")
        # Find corresponding model file
        model_file = output_dir / f"best_model_{dataset_name}_neuron_{neuron_id}_config_{config_name}.npz"
        model_exists = model_file.exists()
        print(f"Model exists == {model_exists}")

        # Extract configuration info
        config = metadata['config']
        regularizer = config.get('regularizer', 'Unknown')
        observation_model = config.get('observation_model', 'Unknown')
        basis_cls_name = config.get('basis_cls_name', 'Unknown')

        # Extract CV scores
        best_score = float(cv_data['best_score'])
        best_index = int(cv_data['best_index'])
        mean_test_scores = cv_data['mean_test_score']
        std_test_scores = cv_data['std_test_score']
        mean_fit_times = cv_data['mean_fit_time']
        std_fit_times = cv_data['std_fit_time']

        # Get best regularization strength if available
        best_reg_strength = None
        if 'param_regularizer_strength' in cv_data:
            reg_strengths = cv_data['param_regularizer_strength']
            if reg_strengths[best_index] is not None:
                best_reg_strength = float(reg_strengths[best_index])

        # Create result entry
        result = {
            # Identifiers
            'dataset_name': dataset_name,
            'neuron_id': int(neuron_id),
            'config_name': config_name,

            # Model configuration
            'regularizer': regularizer,
            'observation_model': observation_model,
            'basis_cls_name': basis_cls_name,

            # Best model performance
            'best_test_score': best_score,
            'best_reg_strength': best_reg_strength,

            # All CV scores (arrays)
            'mean_test_scores': mean_test_scores.tolist(),
            'std_test_scores': std_test_scores.tolist(),
            'mean_fit_times': mean_fit_times.tolist(),
            'std_fit_times': std_fit_times.tolist(),

            # Best model statistics
            'best_test_score_std': float(std_test_scores[best_index]),
            'best_fit_time_mean': float(mean_fit_times[best_index]),
            'best_fit_time_std': float(std_fit_times[best_index]),

            # Metadata
            'binsize': metadata.get('binsize'),
            'history_window': metadata.get('history_window'),
            'n_basis_funcs': metadata.get('n_basis_funcs'),
            'dataset_path': metadata.get('dataset_path'),

            # File paths
            'cv_results_file': cv_file.as_posix(),
            'metadata_file': metadata_file.as_posix(),
            'model_file': model_file.as_posix() if model_exists else None,
            'model_exists': model_exists,

            # Number of parameter combinations tested
            'n_param_combinations': len(mean_test_scores)
        }

        results.append(result)


    print("Finished collection")
    print(results[:3])
    if not results:
        print("Warning: No valid results found!")

        # Let's also check if there are any metadata files at all
        metadata_files = list(output_dir.glob("metadata_*.json"))
        print(f"Found {len(metadata_files)} metadata files")

        if len(metadata_files) > 0:
            print("Some jobs may still be running. Try running this script again later.")

        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by best test score (descending)
    df = df.sort_values('best_test_score', ascending=False).reset_index(drop=True)

    print(f"Successfully collected {len(df)} fit results")

    return df


def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics grouped by model configuration."""

    if df.empty:
        return pd.DataFrame()

    # Group by regularizer and observation model
    summary = df.groupby(['regularizer', 'observation_model', 'basis_cls_name']).agg({
        'best_test_score': ['count', 'mean', 'std', 'min', 'max'],
        'best_reg_strength': ['mean', 'std'],
        'best_fit_time_mean': ['mean', 'std'],
        'n_param_combinations': 'mean'
    }).round(6)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    return summary


def main():
    """Main function to process results and save DataFrame."""

    # Set your output directory path here
    output_dir = pathlib.Path("/mnt/home/ebalzani/ceph/synaptic_connectivity/outputs")

    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        return

    # Collect results
    print("Starting results collection...")
    df = collect_fit_results(output_dir)

    if df.empty:
        print("Error: No results collected!")
        return

    # Save main results DataFrame
    results_csv = output_dir / "all_fit_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved detailed results to {results_csv}")

    # Create and save summary statistics
    summary_df = create_summary_stats(df)
    if not summary_df.empty:
        summary_csv = output_dir / "fit_results_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary statistics to {summary_csv}")

    # Print basic info
    print(f"\nResults Summary:")
    print(f"Total fits collected: {len(df)}")
    print(f"Unique datasets: {df['dataset_name'].nunique()}")
    print(f"Unique neurons: {df['neuron_id'].nunique()}")
    print(f"Unique configurations: {df['config_name'].nunique()}")
    print(f"\nTop 5 performing models:")
    print(df[['config_name', 'dataset_name', 'neuron_id', 'regularizer',
              'observation_model', 'best_test_score', 'best_reg_strength']].head())

    print(f"\nFiles saved:")
    print(f"- Detailed results: {results_csv}")
    print(f"- Summary stats: {summary_csv}")


if __name__ == "__main__":
    main()