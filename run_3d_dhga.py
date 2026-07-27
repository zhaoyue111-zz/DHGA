from __future__ import annotations

import argparse
import json
from pathlib import Path

from dhga.config import DHGAConfig, parse_int_list
from dhga.trainer import run_synthetic_smoke
from voxtell_sfda.adapter import load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DHGA VoxTell SFDA")
    parser.add_argument("--config_json", default="")
    parser.add_argument("--smoke_test", action="store_true", help="Run synthetic random-tensor smoke test only")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and print planned modules without training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default=".save/dhga")
    parser.add_argument("--prompts", nargs="*", default=["liver"])
    parser.add_argument("--dhga_stage", choices=("A", "B", "C", "D"), default="B")
    parser.add_argument("--dhga_freeze_voxtell", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dhga_semantic_adapter_rank", type=int, default=4)
    parser.add_argument("--dhga_semantic_adapter_target", choices=("cross", "self", "both"), default="cross")
    parser.add_argument("--dhga_appearance_feature_layers", nargs="*", default=["-1", "-2", "-3"])
    parser.add_argument("--dhga_appearance_hidden_ratio", type=float, default=0.25)
    parser.add_argument("--dhga_appearance_feature_dropout", type=float, default=0.05)
    parser.add_argument("--dhga_use_ema_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dhga_ema_decay", type=float, default=0.99)
    parser.add_argument("--dhga_anchor_weight", type=float, default=0.25)
    parser.add_argument("--dhga_cross_supervision_weight", type=float, default=1.0)
    parser.add_argument("--dhga_weak_strong_weight", type=float, default=0.5)
    parser.add_argument("--dhga_router_normalization", choices=("case_rank", "none"), default="case_rank")
    parser.add_argument("--dhga_geometry_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dhga_search_radius_mm", type=float, default=6.0)
    parser.add_argument("--dhga_ray_step_mm", type=float, default=1.0)
    parser.add_argument("--dhga_max_boundary_points", type=int, default=4096)
    parser.add_argument("--dhga_boundary_chunk_size", type=int, default=1024)
    parser.add_argument("--dhga_corruption_max_offset_mm", type=float, default=3.0)
    parser.add_argument("--dhga_corruption_modes", nargs="*", default=["inward", "outward"])
    parser.add_argument("--dhga_boundary_recovery_weight", type=float, default=1.0)
    parser.add_argument("--dhga_transport_equivariance_weight", type=float, default=0.1)
    parser.add_argument("--dhga_prompt_ranking_weight", type=float, default=0.05)
    parser.add_argument("--dhga_minimal_transport_weight", type=float, default=0.01)
    parser.add_argument("--dhga_transport_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--dhga_debug_outputs", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DHGAConfig:
    if args.config_json:
        return DHGAConfig.from_json(args.config_json)
    return DHGAConfig(
        dhga_stage=args.dhga_stage,
        dhga_freeze_voxtell=args.dhga_freeze_voxtell,
        dhga_semantic_adapter_rank=args.dhga_semantic_adapter_rank,
        dhga_semantic_adapter_target=args.dhga_semantic_adapter_target,
        dhga_appearance_feature_layers=parse_int_list(args.dhga_appearance_feature_layers, [-1, -2, -3]),
        dhga_appearance_hidden_ratio=args.dhga_appearance_hidden_ratio,
        dhga_appearance_feature_dropout=args.dhga_appearance_feature_dropout,
        dhga_use_ema_teacher=args.dhga_use_ema_teacher,
        dhga_ema_decay=args.dhga_ema_decay,
        dhga_anchor_weight=args.dhga_anchor_weight,
        dhga_cross_supervision_weight=args.dhga_cross_supervision_weight,
        dhga_weak_strong_weight=args.dhga_weak_strong_weight,
        dhga_router_normalization=args.dhga_router_normalization,
        dhga_geometry_enabled=args.dhga_geometry_enabled,
        dhga_search_radius_mm=args.dhga_search_radius_mm,
        dhga_ray_step_mm=args.dhga_ray_step_mm,
        dhga_max_boundary_points=args.dhga_max_boundary_points,
        dhga_boundary_chunk_size=args.dhga_boundary_chunk_size,
        dhga_corruption_max_offset_mm=args.dhga_corruption_max_offset_mm,
        dhga_corruption_modes=list(args.dhga_corruption_modes),
        dhga_boundary_recovery_weight=args.dhga_boundary_recovery_weight,
        dhga_transport_equivariance_weight=args.dhga_transport_equivariance_weight,
        dhga_prompt_ranking_weight=args.dhga_prompt_ranking_weight,
        dhga_minimal_transport_weight=args.dhga_minimal_transport_weight,
        dhga_transport_smoothness_weight=args.dhga_transport_smoothness_weight,
        dhga_debug_outputs=args.dhga_debug_outputs,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    config.validate()
    prompts = load_prompts(args.prompts) if args.prompts else []
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "dhga_config.json").write_text(json.dumps(config.to_dict(), indent=2))

    if args.smoke_test:
        result = run_synthetic_smoke(config, args.device)
        print(json.dumps({
            "smoke_loss": result.loss,
            "shared_encoder_calls": result.shared_encoder_calls,
            "prompts": prompts,
            "diagnostics": result.diagnostics,
            "trainable_summary": result.trainable_summary,
        }, indent=2))
        return

    if args.dry_run:
        print(json.dumps({"config": config.to_dict(), "prompts": prompts}, indent=2))
        return

    raise SystemExit(
        "Real DHGA experiments are intentionally not launched by Codex. "
        "Use DHGA_COMMANDS.md after running --smoke_test and reviewing paths."
    )


if __name__ == "__main__":
    main()
