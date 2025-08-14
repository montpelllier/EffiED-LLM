"""
Label Propagation Experiment
Test label propagation using ground truth labels for error detection
"""
import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import evaluation
from data.loader import DatasetLoader
from data.preprocessing import preprocess_dataset
from detection import FeatureExtractor
from detection.propagation import LabelPropagator


def run_label_propagation_experiment(dataset_name: str, sample_size: int = 20, seed: int = 114,
                                   verbose: bool = False, save_results: bool = False):
    """
    Run label propagation experiment on specified dataset

    Args:
        dataset_name: Name of dataset to test
        sample_size: Number of clusters for label propagation
        seed: Random seed for reproducible results
        verbose: Whether to print detailed progress
        save_results: Whether to save results to file

    Returns:
        Dictionary containing experiment results
    """
    print(f'Loading dataset: {dataset_name}')

    # Load and preprocess dataset
    start_time = time.time()
    datasetloader = DatasetLoader()
    dataset = datasetloader.load_dataset(dataset_name)
    dirty_data, clean_data, err_labels, stats = preprocess_dataset(
        dataset['dirty_data'], dataset['clean_data'], dataset_name, verbose
    )

    if verbose:
        print(f"Dataset shape: {dirty_data.shape}")
        print(f"Error rate: {err_labels.sum().sum() / (err_labels.shape[0] * err_labels.shape[1]):.4f}")

    # Extract features
    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_dataset_feature(dirty_data)

    if verbose:
        print(f"Feature extraction completed. Features for {len(features)} columns.")

    # Label propagation
    print(f"Running label propagation (clusters={sample_size}, seed={seed})...")
    propagator = LabelPropagator(features, sample_size=sample_size, seed=seed)
    prediction_df = propagator.propagate_dataset_labels(err_labels)

    # Evaluation
    print("Evaluating results...")
    evaluator = evaluation.Evaluator()
    result = evaluator.evaluate_detection_results(
        y_true=err_labels,
        y_pred=prediction_df,
        dataset_name=dataset_name,
        model_name=f'RealLabel(k={sample_size})'
    )

    # Add timing information
    total_time = time.time() - start_time
    result['execution_time'] = total_time

    # Print results
    evaluator.print_results(result)
    print(f"\nExecution time: {total_time:.2f} seconds")

    # Save results if requested
    if save_results:
        save_experiment_results(result, dataset_name, sample_size, seed)

    return result


def save_experiment_results(result: dict, dataset_name: str, sample_size: int, seed: int):
    """Save experiment results to file"""
    import json
    from datetime import datetime

    # Create results directory structure
    results_dir = Path(__file__).parent.parent / "results" / "label_propagation"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_k{sample_size}_seed{seed}_{timestamp}.json"
    filepath = results_dir / filename

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    result_clean = convert_numpy_types(result)

    with open(filepath, 'w') as f:
        json.dump(result_clean, f, indent=2)

    print(f"Results saved to: {filepath}")


def run_multiple_experiments():
    """Run experiments on multiple datasets with different configurations"""
    datasets = ['hospital', 'flights', 'beers', 'spotify']
    cluster_sizes = [10, 20, 30]
    seeds = [114, 42, 123]

    all_results = []

    for dataset in datasets:
        for k in cluster_sizes:
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"Experiment: {dataset}, k={k}, seed={seed}")
                print(f"{'='*60}")

                try:
                    result = run_label_propagation_experiment(
                        dataset_name=dataset,
                        sample_size=k,
                        seed=seed,
                        verbose=False,
                        save_results=True
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in experiment {dataset}-k{k}-seed{seed}: {e}")
                    continue

    # Summary of all experiments
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        dataset = result.get('dataset', 'unknown')
        model = result.get('model', 'unknown')
        f1 = result.get('overall', {}).get('f1_score', 0)
        time_taken = result.get('execution_time', 0)
        print(f"{dataset:10} {model:20} F1={f1:.4f} Time={time_taken:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Real Label Propagation Experiment')
    parser.add_argument('--dataset', type=str, default='hospital',
                       choices=['hospital', 'flights', 'beers', 'spotify'],
                       help='Dataset to use for experiment')
    parser.add_argument('--clusters', type=int, default=20,
                       help='Number of clusters for label propagation')
    parser.add_argument('--seed', type=int, default=114,
                       help='Random seed for reproducible results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--multi', action='store_true',
                       help='Run experiments on multiple configurations')

    args = parser.parse_args()

    if args.multi:
        run_multiple_experiments()
    else:
        run_label_propagation_experiment(
            dataset_name=args.dataset,
            sample_size=args.clusters,
            seed=args.seed,
            verbose=args.verbose,
            save_results=args.save
        )


if __name__ == '__main__':
    main()
