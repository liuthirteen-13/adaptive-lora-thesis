from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rank_search.importance import (
    assign_suggested_ranks,
    collect_lora_importance,
    save_importance_json,
)


def test_collect_lora_importance_smoke_keeps_parameters_unchanged() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.utils.data import DataLoader

    class FakeLoraLinear(nn.Module):
        """测试用极小 LoRA 层，模拟 PEFT 注入后的 lora_A/lora_B 结构。"""

        def __init__(self, hidden_size: int, rank: int) -> None:
            super().__init__()
            self.base_layer = nn.Linear(hidden_size, hidden_size, bias=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(hidden_size, rank, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, hidden_size, bias=False)})
            for parameter in self.base_layer.parameters():
                parameter.requires_grad = False

        def forward(self, hidden_states):
            lora_delta = self.lora_B["default"](self.lora_A["default"](hidden_states))
            return self.base_layer(hidden_states) + lora_delta

    class TinySelfAttention(nn.Module):
        def __init__(self, hidden_size: int, rank: int) -> None:
            super().__init__()
            self.q_proj = FakeLoraLinear(hidden_size, rank)

        def forward(self, hidden_states):
            return self.q_proj(hidden_states)

    class TinyLayer(nn.Module):
        def __init__(self, hidden_size: int, rank: int) -> None:
            super().__init__()
            self.self_attn = TinySelfAttention(hidden_size, rank)

        def forward(self, hidden_states):
            return torch.tanh(self.self_attn(hidden_states))

    class TinyBackbone(nn.Module):
        def __init__(self, hidden_size: int, rank: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(hidden_size, rank)])

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    class TinyCausalLM(nn.Module):
        """只实现 importance 需要的 CausalLM 前向接口。"""

        def __init__(self, vocab_size: int = 8, hidden_size: int = 6, rank: int = 2) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.model = TinyBackbone(hidden_size, rank)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, labels=None):
            hidden_states = self.embed(input_ids)
            logits = self.lm_head(self.model(hidden_states))
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(0)
    model = TinyCausalLM()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    dataloader = DataLoader(
        [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([2, 3, 4]), "labels": torch.tensor([3, 4, 5])},
        ],
        batch_size=1,
    )

    importance = collect_lora_importance(model, dataloader, num_batches=2)

    key = "model.layers.0.self_attn.q_proj"
    assert key in importance
    assert importance[key]["grad_norm"] > 0
    assert importance[key]["suggested_rank"] == 2
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter.detach())


def test_assign_suggested_ranks_and_save_json() -> None:
    importance = {
        "model.layers.0.self_attn.q_proj": {"grad_norm": 1.0, "suggested_rank": 4},
        "model.layers.1.self_attn.q_proj": {"grad_norm": 4.0, "suggested_rank": 4},
    }

    adjusted = assign_suggested_ranks(importance, min_rank=2, max_rank=8, rank_step=2)
    assert adjusted["model.layers.0.self_attn.q_proj"]["suggested_rank"] in {2, 4, 6, 8}
    assert adjusted["model.layers.1.self_attn.q_proj"]["suggested_rank"] >= adjusted[
        "model.layers.0.self_attn.q_proj"
    ]["suggested_rank"]

    output_dir = Path(".pytest_cache") / "importance"
    output_file = output_dir / "importance.json"
    try:
        saved_file = save_importance_json(adjusted, output_file)
        assert saved_file.read_text(encoding="utf-8").startswith("{")
    finally:
        output_file.unlink(missing_ok=True)
