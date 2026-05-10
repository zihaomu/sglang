import json
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.observability import (
    log_moe_runtime,
    reset_moe_runtime_log_state,
)


def _moe_runtime_payloads(caplog):
    payloads = []
    for record in caplog.records:
        if record.name == "sglang.srt.layers.moe.observability":
            assert record.message.startswith("moe_runtime ")
            payloads.append(json.loads(record.message.removeprefix("moe_runtime ")))
    return payloads


def test_moe_runtime_logging_is_disabled_by_default(caplog):
    reset_moe_runtime_log_state()
    caplog.set_level(logging.INFO, logger="sglang.srt.layers.moe.observability")

    with envs.SGLANG_LOG_MOE_RUNTIME.override(False):
        log_moe_runtime(
            "unit_event",
            runner_config=MoeRunnerConfig(layer_id=7, num_experts=4, top_k=2),
            runner_backend="triton",
            hidden_states=torch.zeros((2, 8)),
            topk_ids=torch.tensor([[0, 1], [2, 3]]),
        )

    assert _moe_runtime_payloads(caplog) == []


def test_moe_runtime_logging_summarizes_topk_assignments(caplog):
    reset_moe_runtime_log_state()
    caplog.set_level(logging.INFO, logger="sglang.srt.layers.moe.observability")

    with (
        envs.SGLANG_LOG_MOE_RUNTIME.override(True),
        envs.SGLANG_LOG_MOE_RUNTIME_INTERVAL.override(1),
        envs.SGLANG_LOG_MOE_RUNTIME_TOP_EXPERTS.override(2),
        envs.SGLANG_LOG_MOE_RUNTIME_RANKS.override(""),
    ):
        log_moe_runtime(
            "unit_event",
            runner_config=MoeRunnerConfig(
                layer_id=3,
                num_experts=4,
                num_local_experts=2,
                hidden_size=8,
                top_k=2,
            ),
            runner_backend="triton",
            a2a_backend="none",
            fused_path=True,
            hidden_states=torch.zeros((3, 8), dtype=torch.bfloat16),
            topk_ids=torch.tensor([[0, 1], [1, 3], [-1, 99]]),
            topk_weights=torch.ones((3, 2), dtype=torch.float32),
            w1=torch.zeros((4, 16, 8), dtype=torch.bfloat16),
            w2=torch.zeros((4, 8, 8), dtype=torch.bfloat16),
            quant={"use_fp8_w8a8": False},
            kernel={"BLOCK_SIZE_M": 64},
        )

    payloads = _moe_runtime_payloads(caplog)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["event"] == "unit_event"
    assert payload["runner_config"]["layer_id"] == 3
    assert payload["shapes"]["hidden_states"] == [3, 8]
    assert payload["dtypes"]["hidden_states"] == "torch.bfloat16"
    assert payload["topk"]["valid_assignments"] == 4
    assert payload["topk"]["invalid_assignments"] == 2
    assert payload["topk"]["top_experts"][0] == {"expert": 1, "count": 2}
    assert payload["quant"] == {"use_fp8_w8a8": False}
    assert payload["kernel"] == {"BLOCK_SIZE_M": 64}


def test_moe_runtime_logging_respects_interval(caplog):
    reset_moe_runtime_log_state()
    caplog.set_level(logging.INFO, logger="sglang.srt.layers.moe.observability")

    with (
        envs.SGLANG_LOG_MOE_RUNTIME.override(True),
        envs.SGLANG_LOG_MOE_RUNTIME_INTERVAL.override(2),
        envs.SGLANG_LOG_MOE_RUNTIME_RANKS.override(""),
    ):
        config = MoeRunnerConfig(layer_id=5, num_experts=2, top_k=1)
        for _ in range(3):
            log_moe_runtime(
                "unit_event",
                runner_config=config,
                runner_backend="aiter",
                topk_ids=torch.tensor([[0], [1]]),
            )

    payloads = _moe_runtime_payloads(caplog)
    assert [payload["event"] for payload in payloads] == ["unit_event", "unit_event"]
