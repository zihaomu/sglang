from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_log_counters: defaultdict[tuple[str, Optional[int], str], int] = defaultdict(int)


def moe_runtime_logging_enabled() -> bool:
    return envs.SGLANG_LOG_MOE_RUNTIME.get()


def reset_moe_runtime_log_state() -> None:
    _log_counters.clear()


def log_moe_runtime(
    event: str,
    *,
    runner_config: Optional[Any] = None,
    runner_backend: Optional[str] = None,
    a2a_backend: Optional[str] = None,
    fused_path: Optional[bool] = None,
    dispatch_format: Optional[str] = None,
    hidden_states: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    w1: Optional[torch.Tensor] = None,
    w2: Optional[torch.Tensor] = None,
    quant: Optional[dict[str, Any]] = None,
    kernel: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    if not _should_log(event, _get_int_attr(runner_config, "layer_id"), runner_backend):
        return

    num_experts = _get_int_attr(runner_config, "num_experts")
    data: dict[str, Any] = {
        "event": event,
        "rank": _rank(),
        "local_rank": _local_rank(),
        "runner_backend": runner_backend,
        "a2a_backend": a2a_backend,
        "fused_path": fused_path,
        "dispatch_format": dispatch_format,
        "runner_config": _runner_config_payload(runner_config),
        "shapes": {
            "hidden_states": _tensor_shape(hidden_states),
            "topk_ids": _tensor_shape(topk_ids),
            "topk_weights": _tensor_shape(topk_weights),
            "w1": _tensor_shape(w1),
            "w2": _tensor_shape(w2),
        },
        "dtypes": {
            "hidden_states": _tensor_dtype(hidden_states),
            "topk_ids": _tensor_dtype(topk_ids),
            "topk_weights": _tensor_dtype(topk_weights),
            "w1": _tensor_dtype(w1),
            "w2": _tensor_dtype(w2),
        },
        "topk": _safe_topk_payload(topk_ids, num_experts),
        "quant": _jsonable(quant or {}),
        "kernel": _jsonable(kernel or {}),
        "extra": _jsonable(extra or {}),
    }
    logger.info("moe_runtime %s", json.dumps(data, sort_keys=True))


def _should_log(
    event: str, layer_id: Optional[int], runner_backend: Optional[str]
) -> bool:
    if not moe_runtime_logging_enabled():
        return False
    if not _rank_is_enabled():
        return False

    interval = envs.SGLANG_LOG_MOE_RUNTIME_INTERVAL.get()
    if interval is None or interval <= 0:
        interval = 1

    key = (event, layer_id, runner_backend or "")
    counter = _log_counters[key]
    _log_counters[key] = counter + 1
    return counter % interval == 0


def _rank_is_enabled() -> bool:
    ranks = envs.SGLANG_LOG_MOE_RUNTIME_RANKS.get()
    if not ranks:
        return True

    enabled_ranks: set[int] = set()
    for item in ranks.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            enabled_ranks.add(int(item))
        except ValueError:
            logger.warning("Invalid SGLANG_LOG_MOE_RUNTIME_RANKS entry: %s", item)
            return True
    return _rank() in enabled_ranks


def _rank() -> int:
    return _env_int("RANK", 0)


def _local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _runner_config_payload(runner_config: Optional[Any]) -> dict[str, Any]:
    if runner_config is None:
        return {}

    keys = [
        "layer_id",
        "num_experts",
        "num_local_experts",
        "hidden_size",
        "intermediate_size_per_partition",
        "top_k",
        "num_fused_shared_experts",
        "routing_method_type",
        "activation",
        "is_gated",
        "apply_router_weight_on_input",
        "inplace",
        "no_combine",
        "routed_scaling_factor",
        "gemm1_alpha",
        "gemm1_clamp_limit",
        "swiglu_limit",
    ]
    payload = {key: _jsonable(getattr(runner_config, key, None)) for key in keys}
    payload["params_dtype"] = _jsonable(getattr(runner_config, "params_dtype", None))
    return payload


def _safe_topk_payload(
    topk_ids: Optional[torch.Tensor], num_experts: Optional[int]
) -> Optional[dict[str, Any]]:
    try:
        return _topk_payload(topk_ids, num_experts)
    except Exception as err:
        return {"error": str(err)}


def _topk_payload(
    topk_ids: Optional[torch.Tensor], num_experts: Optional[int]
) -> Optional[dict[str, Any]]:
    if topk_ids is None or topk_ids.numel() == 0:
        return None

    top_experts = envs.SGLANG_LOG_MOE_RUNTIME_TOP_EXPERTS.get()
    if top_experts is None or top_experts <= 0:
        top_experts = 1

    with torch.no_grad():
        flat_ids = topk_ids.detach().reshape(-1).to(torch.int64)
        valid_mask = flat_ids >= 0
        if num_experts is not None and num_experts > 0:
            valid_mask &= flat_ids < num_experts
        valid_ids = flat_ids[valid_mask]

        total_assignments = int(flat_ids.numel())
        valid_assignments = int(valid_ids.numel())
        if valid_assignments == 0:
            return {
                "total_assignments": total_assignments,
                "valid_assignments": 0,
                "invalid_assignments": total_assignments,
                "nonzero_experts": 0,
                "top_experts": [],
            }

        minlength = num_experts if num_experts is not None and num_experts > 0 else 0
        counts = torch.bincount(valid_ids, minlength=minlength)
        top_count = min(int(top_experts), int(counts.numel()))
        top_values, top_indices = torch.topk(counts, top_count)
        top_values_list = [int(v) for v in top_values.cpu().tolist()]
        top_indices_list = [int(i) for i in top_indices.cpu().tolist()]
        nonzero_experts = int(torch.count_nonzero(counts).cpu().item())

    payload: dict[str, Any] = {
        "total_assignments": total_assignments,
        "valid_assignments": valid_assignments,
        "invalid_assignments": total_assignments - valid_assignments,
        "nonzero_experts": nonzero_experts,
        "top_experts": [
            {"expert": expert, "count": count}
            for expert, count in zip(top_indices_list, top_values_list)
            if count > 0
        ],
    }

    if num_experts is not None and num_experts > 0:
        average = valid_assignments / num_experts
        payload["max_to_average"] = (
            round(top_values_list[0] / average, 4) if average > 0 else None
        )

    return payload


def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[list[int]]:
    if tensor is None:
        return None
    return [int(dim) for dim in tensor.shape]


def _tensor_dtype(tensor: Optional[torch.Tensor]) -> Optional[str]:
    if tensor is None:
        return None
    return str(tensor.dtype)


def _get_int_attr(obj: Optional[Any], name: str) -> Optional[int]:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": _tensor_shape(value),
            "dtype": _tensor_dtype(value),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
