import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from tools.trainers.endodepth import plEndoDepth
from datasets.blender_dataset import BlenderDataset
from networks.metrics import compute_depth_metrics
from networks import layers


def evaluate(checkpoint_path, hparams_path, data_path, batch_size=16):
    """
    Evaluate trained model on validation dataset and compute depth metrics.
    
    Args:
        checkpoint_path: Path to model checkpoint (.ckpt file)
        hparams_path: Path to hyperparameters YAML file
        data_path: Path to dataset root directory
        batch_size: Batch size for evaluation
    """
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Load model
    model = plEndoDepth.load_from_checkpoint(
        checkpoint_path,
        test=True,
        hparams_file=hparams_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Load validation dataset
    print(f"Loading validation dataset from: {data_path}")
    # BlenderDataset loads all images from the data_path
    # We'll use all available data for evaluation
    val_dataset = BlenderDataset(path=data_path)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Starting evaluation...")
    
    # Accumulate metrics
    all_metrics = {
        "de/abs_rel": [],
        "de/sq_rel": [],
        "de/rms": [],
        "de/log_rms": [],
        "da/a1": [],
        "da/a2": [],
        "da/a3": [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Get inputs
            images = batch["color"].squeeze(1).to(device)
            depth_gt = batch["depth"].squeeze(1).to(device)
            
            # Forward pass - model already returns depth when in eval mode
            depth_pred = model.forward(images)
            
            # Filter out invalid depth values (zeros or very small values)
            # This prevents division by zero and log(0) issues
            valid_mask = depth_gt > 1e-3
            if valid_mask.sum() == 0:
                continue  # Skip this batch if all values are invalid
            
            depth_gt_valid = depth_gt[valid_mask]
            depth_pred_valid = depth_pred[valid_mask]
            
            # Compute metrics for this batch
            batch_metrics = compute_depth_metrics(depth_gt_valid, depth_pred_valid)
            
            # Accumulate
            for key in all_metrics.keys():
                all_metrics[key].append(batch_metrics[key].item())
    
    # Compute mean metrics
    mean_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    
    return mean_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth estimation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--hparams', type=str, required=True, help='Path to hparams.yaml file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        data_path=args.data_path,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Abs.Rel:  {metrics['de/abs_rel']:.4f}")
    print(f"Sq.Rel:   {metrics['de/sq_rel']:.4f}")
    print(f"RMSE:     {metrics['de/rms']:.4f}")
    print(f"RMSElog:  {metrics['de/log_rms']:.4f}")
    print(f"δ₁ (a1):  {metrics['da/a1']:.4f}")
    print(f"δ₂ (a2):  {metrics['da/a2']:.4f}")
    print(f"δ₃ (a3):  {metrics['da/a3']:.4f}")
    print("="*60)
    
    # Compare with paper results (from README and PAPER_COMPARISON.md)
    paper_metrics = {
        "Abs.Rel": 0.276,
        "RMSE": 0.066,
        "RMSElog": 0.349,
        "δ₁": 0.517,
        "δ₂": 0.819,
        "δ₃": 0.941
    }
    
    print("\nCOMPARISON WITH PAPER")
    print("="*60)
    print(f"{'Metric':<15} {'Paper':<12} {'Ours':<12} {'Diff':<12}")
    print("-"*60)
    print(f"{'Abs.Rel':<15} {paper_metrics['Abs.Rel']:<12.4f} {metrics['de/abs_rel']:<12.4f} {metrics['de/abs_rel'] - paper_metrics['Abs.Rel']:+.4f}")
    print(f"{'RMSE':<15} {paper_metrics['RMSE']:<12.4f} {metrics['de/rms']:<12.4f} {metrics['de/rms'] - paper_metrics['RMSE']:+.4f}")
    print(f"{'RMSElog':<15} {paper_metrics['RMSElog']:<12.4f} {metrics['de/log_rms']:<12.4f} {metrics['de/log_rms'] - paper_metrics['RMSElog']:+.4f}")
    print(f"{'δ₁ (a1)':<15} {paper_metrics['δ₁']:<12.4f} {metrics['da/a1']:<12.4f} {metrics['da/a1'] - paper_metrics['δ₁']:+.4f}")
    print(f"{'δ₂ (a2)':<15} {paper_metrics['δ₂']:<12.4f} {metrics['da/a2']:<12.4f} {metrics['da/a2'] - paper_metrics['δ₂']:+.4f}")
    print(f"{'δ₃ (a3)':<15} {paper_metrics['δ₃']:<12.4f} {metrics['da/a3']:<12.4f} {metrics['da/a3'] - paper_metrics['δ₃']:+.4f}")
    print("="*60)
    
    # Save to JSON
    output_data = {
        "metrics": metrics,
        "paper_metrics": {
            "de/abs_rel": paper_metrics["Abs.Rel"],
            "de/rms": paper_metrics["RMSE"],
            "de/log_rms": paper_metrics["RMSElog"],
            "da/a1": paper_metrics["δ₁"],
            "da/a2": paper_metrics["δ₂"],
            "da/a3": paper_metrics["δ₃"]
        },
        "checkpoint": args.checkpoint,
        "data_path": args.data_path
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
