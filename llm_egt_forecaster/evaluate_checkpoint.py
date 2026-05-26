import argparse
import json
import os
import random
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_EVAL_DATA_PATH = os.path.join(
    _REPO_ROOT,
    "llm_egt_forecaster",
    "data",
    "real_dataset_100_2026_test.json",
)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.src.dataset import get_dataloader
from llm_egt_forecaster.src.models.evolutionary_framework import EvolutionaryFramework


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LLM-EGT checkpoint on a JSON dataset.")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_EVAL_DATA_PATH,
        help=(
            "Path to a real_dataset_enhanced-style JSON file. Relative paths are "
            "resolved from the repository root. Defaults to the 2026 holdout test set."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--news-selector-method", choices=["api", "cosine"], default=None)
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics JSON.")
    parser.add_argument("--save-predictions", default=None, help="Optional path to save per-sample predictions JSON.")
    parser.add_argument("--sample-errors-json", default=None, help="Optional path to save per-sample error metrics JSON.")
    parser.add_argument("--sample-errors-log", default=None, help="Optional path to save a readable per-sample error log.")
    return parser.parse_args()


def resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def filter_quantization_state(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    skip_fragments = (
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
        ".quant_state",
        "bitsandbytes",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if not any(fragment in key for fragment in skip_fragments)
    }


def denormalize(values, norm_mean, norm_std):
    eps = 1e-8
    while norm_mean.dim() < values.dim():
        norm_mean = norm_mean.unsqueeze(-1)
        norm_std = norm_std.unsqueeze(-1)
    return values * (norm_std + eps) + norm_mean


def dataset_original_indices(dataloader):
    dataset = dataloader.dataset
    if hasattr(dataset, "indices"):
        return list(dataset.indices)
    return list(range(len(dataset)))


def write_sample_error_log(path, sample_errors):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "eval_index\tglobal_index\tdate\tregion\tmse\trmse\tmae\tmape_percent\tnews_count\n"
        )
        for item in sample_errors:
            metadata = item.get("metadata", {})
            f.write(
                f"{item.get('eval_index')}\t"
                f"{item.get('global_index')}\t"
                f"{metadata.get('date', '')}\t"
                f"{metadata.get('region', '')}\t"
                f"{item['mse']:.6f}\t"
                f"{item['rmse']:.6f}\t"
                f"{item['mae']:.6f}\t"
                f"{item['mape_percent']:.6f}\t"
                f"{item.get('news_count', 0)}\n"
            )


def main():
    args = parse_args()
    config = base_config

    args.data_path = resolve_repo_path(args.data_path)
    args.checkpoint = resolve_repo_path(args.checkpoint)
    args.output_json = resolve_repo_path(args.output_json)
    args.save_predictions = resolve_repo_path(args.save_predictions)
    args.sample_errors_json = resolve_repo_path(args.sample_errors_json)
    args.sample_errors_log = resolve_repo_path(args.sample_errors_log)

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_agents is not None:
        config.NUM_AGENTS = args.num_agents
    if args.news_selector_method is not None:
        config.NEWS_SELECTOR_METHOD = args.news_selector_method

    set_seed(config.SEED)

    print("=" * 80)
    print("LLM-EGT Checkpoint Evaluation")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {config.DEVICE}")
    print(f"Base LLM: {config.BASE_LLM_MODEL}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Split: {args.split}")
    print(f"News selector: {config.NEWS_SELECTOR_METHOD}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = get_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        config=config,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Evaluation samples: {len(dataloader.dataset)}")

    framework = EvolutionaryFramework(config).to(config.DEVICE)

    checkpoint = torch_load_checkpoint(args.checkpoint, config.DEVICE)
    framework_state = filter_quantization_state(checkpoint.get("framework_state_dict", checkpoint))
    missing, unexpected = framework.load_state_dict(framework_state, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if "agent_logics" in checkpoint:
        for i, agent in enumerate(framework.agents):
            if i < len(checkpoint["agent_logics"]):
                agent.logic = checkpoint["agent_logics"][i]
            if "agent_fitness" in checkpoint and i < len(checkpoint["agent_fitness"]):
                agent.fitness = checkpoint["agent_fitness"][i]
        print("Restored agent logics/fitness from checkpoint.")

    framework.eval()

    total_sse = 0.0
    total_ae = 0.0
    total_ape = 0.0
    total_points = 0

    total_sse_norm = 0.0
    total_ae_norm = 0.0

    total_forward_s = 0.0
    total_samples = 0
    running_sample_idx = 0
    original_indices = dataset_original_indices(dataloader)
    prediction_records = []
    sample_errors = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_s = time.perf_counter()
            output = framework(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_forward_s += time.perf_counter() - start_s

            pred_norm = output["aggregated_prediction"]
            gt_norm = batch["ground_truth"].to(config.DEVICE)
            norm_mean = batch["norm_mean"].to(config.DEVICE)
            norm_std = batch["norm_std"].to(config.DEVICE)

            pred = denormalize(pred_norm, norm_mean, norm_std)
            gt = denormalize(gt_norm, norm_mean, norm_std)

            err = pred - gt
            abs_err = torch.abs(err)
            abs_pct_err = abs_err / (torch.abs(gt) + 1e-8)

            total_sse += torch.sum(err ** 2).item()
            total_ae += torch.sum(abs_err).item()
            total_ape += torch.sum(abs_pct_err).item()
            total_points += gt.numel()

            norm_err = pred_norm - gt_norm
            total_sse_norm += torch.sum(norm_err ** 2).item()
            total_ae_norm += torch.sum(torch.abs(norm_err)).item()
            total_samples += len(batch["time_series_str"])

            per_sample_sse = torch.sum(err ** 2, dim=1)
            per_sample_mse = per_sample_sse / gt.shape[1]
            per_sample_rmse = torch.sqrt(per_sample_mse)
            per_sample_mae = torch.mean(abs_err, dim=1)
            per_sample_mape = torch.mean(abs_pct_err, dim=1) * 100.0

            metadata = batch.get("metadata", [{} for _ in range(gt.shape[0])])
            candidate_news = batch.get("candidate_news", [[] for _ in range(gt.shape[0])])
            for i in range(gt.shape[0]):
                global_index = original_indices[running_sample_idx] if running_sample_idx < len(original_indices) else running_sample_idx
                news_count = len(candidate_news[i]) if i < len(candidate_news) and isinstance(candidate_news[i], list) else 0
                sample_errors.append({
                    "eval_index": running_sample_idx,
                    "global_index": int(global_index),
                    "batch_index": batch_idx,
                    "sample_index_in_batch": i,
                    "metadata": metadata[i] if i < len(metadata) else {},
                    "news_count": news_count,
                    "mse": float(per_sample_mse[i].item()),
                    "rmse": float(per_sample_rmse[i].item()),
                    "mae": float(per_sample_mae[i].item()),
                    "mape_percent": float(per_sample_mape[i].item()),
                })
                running_sample_idx += 1

            if args.save_predictions:
                pred_cpu = pred.detach().cpu().tolist()
                gt_cpu = gt.detach().cpu().tolist()
                for i in range(len(pred_cpu)):
                    record_idx = len(prediction_records)
                    error_item = sample_errors[record_idx] if record_idx < len(sample_errors) else {}
                    prediction_records.append({
                        "batch_index": batch_idx,
                        "sample_index_in_batch": i,
                        "global_index": error_item.get("global_index"),
                        "metadata": metadata[i] if i < len(metadata) else {},
                        "metrics": {
                            "mse": error_item.get("mse"),
                            "rmse": error_item.get("rmse"),
                            "mae": error_item.get("mae"),
                            "mape_percent": error_item.get("mape_percent"),
                        },
                        "prediction": pred_cpu[i],
                        "ground_truth": gt_cpu[i],
                    })

    mse = total_sse / total_points
    rmse = mse ** 0.5
    mae = total_ae / total_points
    mape = (total_ape / total_points) * 100.0

    mse_norm = total_sse_norm / total_points
    rmse_norm = mse_norm ** 0.5
    mae_norm = total_ae_norm / total_points

    metrics = {
        "num_samples": total_samples,
        "num_points": total_points,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
        "mse_normalized": mse_norm,
        "rmse_normalized": rmse_norm,
        "mae_normalized": mae_norm,
        "latency_ms_per_batch": (total_forward_s / max(1, len(dataloader))) * 1000.0,
        "latency_ms_per_sample": (total_forward_s / max(1, total_samples)) * 1000.0,
    }

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Samples: {metrics['num_samples']}")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape_percent']:.6f}%")
    print("\n[Normalized Space]")
    print(f"MSE:  {metrics['mse_normalized']:.6f}")
    print(f"RMSE: {metrics['rmse_normalized']:.6f}")
    print(f"MAE:  {metrics['mae_normalized']:.6f}")
    print("\n[Efficiency]")
    print(f"Latency: {metrics['latency_ms_per_batch']:.6f} ms/batch")
    print(f"Latency: {metrics['latency_ms_per_sample']:.6f} ms/sample")
    print("=" * 80)

    if args.output_json:
        ensure_parent_dir(args.output_json)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {args.output_json}")

    if args.sample_errors_json:
        ensure_parent_dir(args.sample_errors_json)
        with open(args.sample_errors_json, "w", encoding="utf-8") as f:
            json.dump(sample_errors, f, ensure_ascii=False, indent=2)
        print(f"Saved per-sample errors JSON to: {args.sample_errors_json}")

    if args.sample_errors_log:
        ensure_parent_dir(args.sample_errors_log)
        write_sample_error_log(args.sample_errors_log, sample_errors)
        print(f"Saved per-sample error log to: {args.sample_errors_log}")

    if args.save_predictions:
        ensure_parent_dir(args.save_predictions)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(prediction_records, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions to: {args.save_predictions}")


if __name__ == "__main__":
    main()
