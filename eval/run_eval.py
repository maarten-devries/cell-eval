from eval.metric_evaluator import MetricsEvaluator
import scanpy as sc
import argparse
import yaml

def parse_args():
    """
    CLI for evaluation
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained cell state model.")
    parser.add_argument(
        "--adata_pred",
        type=str,
        default='/home/yhr/state-eval/adata_pred_subset.h5ad',
        help="Path to the predicted adata object to evaluate",
    )
    parser.add_argument(
        "--adata_true",
        type=str,
        default='/home/yhr/state-eval/adata_true_subset.h5ad',
        help="Path to the true adata object to evaluate against",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default='/home/yhr/state-eval/config/replogle_config.yaml',
        help="If set, will load the config.yaml file from the output_dir and use it to set up the model.",
    )
    return parser.parse_args()

def main():
    """
    Main function to run the evaluation.
    """

    # Parse arguments
    args = parse_args()
    print('Reading adata objects')
    # Read the adata objects
    adata_pred = sc.read_h5ad(args.adata_pred)
    adata_real = sc.read_h5ad(args.adata_true)

    # Read in config file
    with open(args.eval_config, "r") as f:
        config = yaml.safe_load(f)

    print('Running evaluation')
    # Create the evaluator
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=config["include_dist_metrics"],
        control_pert=config["control_pert"],
        pert_col=config["pert_col"],
        celltype_col=config["celltype_col"],
        output_space=config["output_space"],
        shared_perts=config["shared_perts"],
        outdir=config["outdir"],
        de_metric=config["de_metric"],
        class_score=config["class_score"]
    )

    # Compute the metrics
    metrics = evaluator.compute()
    evaluator.save_metrics_per_celltype(metrics, average=True)

    print("Done")

if __name__ == "__main__":
    main()
