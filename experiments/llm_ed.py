"""
Main program example
Demonstrates how to use the refactored modules under the src directory
"""
import argparse
import sys
from pathlib import Path

import detection
import evaluation

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data import DatasetLoader, preprocess_dataset
from llm import llm_factory
from detection import FeatureExtractor, LabelPropagator


def run_llm_ed(dataset_name, model_name, verbose, sample_size, seed, strategies, fewshot, batch_size):
    """
    Run LLM-based Error Detection pipeline

    Args:
        dataset_name (str): Name of the dataset to process
        model_name (str): Name of the LLM model to use
        verbose (bool): Whether to print verbose output
        sample_size (int): Number of clusters for label propagation
        seed (int): Random seed for reproducibility
        strategies (list): Detection strategies to use
        fewshot (int): Number of few-shot examples
        batch_size (int): Batch size for processing
    """
    # 1. Data preparation
    print("=== Data Loading ===")
    loader = DatasetLoader()
    dataset = loader.load_dataset(dataset_name)
    preprocess_dataset(dataset, verbose)
    dirty_data, clean_data, err_labels = dataset.dirty_data, dataset.clean_data, dataset.error_labels

    # 2. LLM configuration
    print("=== LLM Configuration ===")
    llm = llm_factory.create_llm_from_config(model_name=model_name)
    print(f"Using model: {llm.get_model_name()}, {type(llm)}")

    # 3. Feature extraction and label propagation
    print("=== Feature Extraction ===")
    extractor = FeatureExtractor()
    features = extractor.extract_dataset_feature(dirty_data)

    print(f"Running label propagation (clusters={sample_size}, seed={seed})...")
    propagator = LabelPropagator(features, sample_size=sample_size, seed=seed)
    cluster_df, repre_df = propagator.sample()

    # 4. Error detection
    print("=== Error Detection ===")
    detector = detection.ErrorDetector(dataset, repre_df, llm, strategies, fewshot, batch_size)
    print(f"Detection configuration: {detector.prompt_config}")
    detection_result = detector.detect_errors()

    # 5. Label propagation
    print("=== Label Propagation ===")
    for column_name, repre_labels in detection_result.items():
        propagator.propagate_column_labels(column_name, repre_labels)

    prediction_df = propagator.get_prediction()

    # 6. Results evaluation
    print("=== Evaluation ===")
    evaluator = evaluation.Evaluator()
    result = evaluator.evaluate_detection_results(
        y_true=err_labels,
        y_pred=prediction_df,
        dataset_name=dataset_name,
        model_name=f'{model_name}(k={sample_size})'
    )
    evaluator.print_results(result)

    return result


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="LLM-based Error Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='beers',
        choices=['beers', 'flights', 'hospital', 'spotify'],
        help='Dataset to use for error detection'
    )

    # Model parameters
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='qwen3:8b',
        help='LLM model name to use for error detection'
    )

    # Detection strategy parameters
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['ZS-R', 'A'],
        choices=['ZS-I', 'ZS-R', 'V', 'A'],
        help='Detection strategies to use (Zero-Shot Instruction/Rule, Value Only/Attributes)'
    )

    parser.add_argument(
        '--fewshot',
        type=int,
        default=0,
        help='Number of few-shot examples to include in prompts'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing data'
    )

    # Label propagation parameters
    parser.add_argument(
        '--sample-size', '-k',
        type=int,
        default=20,
        help='Number of clusters for label propagation'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=114,
        help='Random seed for reproducibility'
    )

    # Output parameters
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Disable verbose output'
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle quiet flag
    if args.quiet:
        args.verbose = False

    # Print configuration
    print("=== Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Strategies: {args.strategies}")
    print(f"Few-shot examples: {args.fewshot}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sample size (clusters): {args.sample_size}")
    print(f"Random seed: {args.seed}")
    print(f"Verbose: {args.verbose}")
    print()

    # Run the error detection pipeline
    try:
        result = run_llm_ed(
            dataset_name=args.dataset,
            model_name=args.model,
            verbose=args.verbose,
            sample_size=args.sample_size,
            seed=args.seed,
            strategies=args.strategies,
            fewshot=args.fewshot,
            batch_size=args.batch_size
        )
        print("\n=== Pipeline completed successfully ===")
        return result
    except Exception as e:
        print(f"\n=== Error occurred: {e} ===")
        raise


if __name__ == "__main__":
    main()
