"""Tests for the Stylizer v4 reference-conditioning changes.

Covers: learned null-reference token, ref_present flag, ref_cross_positions
(each ref-cross block runs exactly once), and legacy (v3) checkpoint loading.

The reference adapters / cross-attention are zero-gated at init, so every
"does the reference matter" test first opens the gates.

Run:  python -m pytest tests/test_ref_conditioning.py -q
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402


def cfg(**over):
    base = dict(
        image_size=32, in_channels=4, patch_size=2, hidden_size=64,
        num_heads=4, mlp_ratio=2.0, text_mlp_ratio=1.0,
        double_stream_blocks=2, single_stream_blocks=4, text_dim=32,
        max_text_tokens=8, text_injection="cross_attn", cross_attn_blocks=2,
        qk_norm=True, rope="2d", activation="swiglu", bias=False,
        cond_dropout=0.1, ref_condition="adapter+cross", ref_size=64,
        ref_channels=4, ref_adapter_positions=[0, 1], ref_cross_blocks=2,
    )
    base.update(over)
    return ModelConfig(**base)


def make_model(open_gates=True, seed=0, **over):
    torch.manual_seed(seed)
    m = MCFlowDiT(cfg(**over)).eval()
    if open_gates:
        with torch.no_grad():
            for p in m.ref_adapters:
                p.gate.fill_(1.0)
            for p in m.ref_cross_blocks:
                p.gate.fill_(1.0)
    return m


def _batch(B=3, S=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 4, 32, 32, generator=g)
    t = torch.rand(B, generator=g)
    text = torch.randn(B, 8, 32, generator=g)
    tmask = torch.ones(B, 8, dtype=torch.bool)
    ref = torch.randn(B, 4, S, S, generator=g)
    return x, t, text, tmask, ref


def test_model_has_learned_null():
    m = MCFlowDiT(cfg())
    assert hasattr(m, "ref_null")
    assert m.ref_null.shape == (64,)


def test_ref_present_false_matches_reference_none():
    """ref_present=False must be identical to feeding no reference at all."""
    m = make_model()
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        v_none = m(x, t, text, text_mask=tmask, reference=None)
        v_flag = m(x, t, text, text_mask=tmask, reference=torch.zeros_like(ref),
                   ref_present=torch.zeros(x.shape[0], dtype=torch.bool))
        v_real = m(x, t, text, text_mask=tmask, reference=ref)
    assert torch.allclose(v_none, v_flag, atol=1e-5), (
        "ref_present=False must reproduce the null-reference branch")
    assert not torch.allclose(v_none, v_real, atol=1e-3), (
        "with gates open, a real reference must change the output")


def test_zero_reference_without_flag_is_not_null():
    """A zero image with ref_present unset is NOT the null branch.

    At init ref_null is 0 and encoder(0) is also 0 (no biases anywhere), so
    the two coincide; once ref_null is learned they must diverge. This is why
    v4 carries an explicit ref_present flag instead of relying on zeros.
    """
    m = make_model()
    with torch.no_grad():
        m.ref_null.fill_(0.7)  # simulate a learned null token
    x, t, text, tmask, ref = _batch()
    z = torch.zeros_like(ref)
    with torch.no_grad():
        v_null = m(x, t, text, text_mask=tmask, reference=None)
        v_zero_flagged = m(x, t, text, text_mask=tmask, reference=z,
                           ref_present=torch.zeros(x.shape[0], dtype=torch.bool))
        v_zero_plain = m(x, t, text, text_mask=tmask, reference=z)
    assert torch.allclose(v_null, v_zero_flagged, atol=1e-5)
    assert not torch.allclose(v_null, v_zero_plain, atol=1e-4), (
        "zeros with present=True must differ from the learned null branch")


def test_ref_present_mixed_batch():
    """Half the batch absent, half present: each sample is independent."""
    m = make_model()
    x, t, text, tmask, ref = _batch(B=4)
    present = torch.tensor([True, False, True, False])
    with torch.no_grad():
        v = m(x, t, text, text_mask=tmask, reference=ref, ref_present=present)
        v0 = m(x[:1], t[:1], text[:1], text_mask=tmask[:1], reference=ref[:1])
        v1 = m(x[1:2], t[1:2], text[1:2], text_mask=tmask[1:2], reference=None)
    assert torch.allclose(v[0:1], v0, atol=1e-5)
    assert torch.allclose(v[1:2], v1, atol=1e-5)


def test_ref_dropout_uses_null_token():
    """Dropout must route to the learned null, not to a zero image."""
    m = make_model()
    with torch.no_grad():
        m.ref_null.fill_(0.7)
    x, t, text, tmask, ref = _batch(B=2)
    with torch.no_grad():
        v_null = m(x, t, text, text_mask=tmask, reference=None)
        v_drop = m(x, t, text, text_mask=tmask, reference=ref,
                   ref_present=torch.zeros(2, dtype=torch.bool))
        v_zero_plain = m(x, t, text, text_mask=tmask, reference=torch.zeros_like(ref))
    assert torch.allclose(v_null, v_drop, atol=1e-5)
    assert not torch.allclose(v_null, v_zero_plain, atol=1e-4)


def _count_ref_cross(m):
    """-> list of per-block call counters via forward hooks."""
    calls = [0] * len(m.ref_cross_blocks)

    def hook_factory(k):
        def hook(_mod, _args, _out):
            calls[k] += 1
        return hook

    for k, blk in enumerate(m.ref_cross_blocks):
        blk.register_forward_hook(hook_factory(k))
    return calls


def test_ref_cross_positions_default_runs_once():
    """Default [-1, -1]: both ref-cross blocks run once, after all blocks."""
    m = make_model(ref_cross_positions=[-1, -1])
    assert m.ref_cross_positions == [-1, -1]
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=ref)
    assert calls == [1, 1], f"default positions must run each block once, got {calls}"


def test_ref_cross_positions_mid_runs_once():
    """Positions [1, -1]: block0 after single-block 1, block1 at the end."""
    m = make_model(ref_cross_positions=[1, -1])
    assert m.ref_cross_positions == [1, -1]
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=ref)
    assert calls == [1, 1], f"each block must run exactly once, got {calls}"


def test_ref_cross_positions_all_mid():
    """Every block mapped to a distinct mid position: still exactly once each."""
    m = make_model(ref_cross_positions=[0, 3])
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=ref)
    assert calls == [1, 1], f"got {calls}"


def test_ref_cross_positions_duplicate_mid():
    """Two blocks on the same position both run (in list order)."""
    m = make_model(ref_cross_positions=[2, 2])
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=ref)
    assert calls == [1, 1], f"got {calls}"


def test_ref_cross_positions_out_of_range_goes_to_tail():
    """A position past the last single block falls back to the tail."""
    m = make_model(ref_cross_positions=[99, -1])
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=ref)
    assert calls == [1, 1], f"got {calls}"


def test_ref_cross_positions_skipped_without_reference():
    """No reference (null token absent) -> ref-cross blocks must not run."""
    m = make_model()
    m.legacy_null = True  # reference=None skips the whole branch
    calls = _count_ref_cross(m)
    x, t, text, tmask, ref = _batch()
    with torch.no_grad():
        m(x, t, text, text_mask=tmask, reference=None)
    assert calls == [0, 0], f"got {calls}"


def test_ref_cross_positions_padded():
    """A short list is padded with -1 so every block has a position."""
    m = MCFlowDiT(cfg(ref_cross_positions=[2]))
    assert m.ref_cross_positions == [2, -1]


def test_legacy_null_flag_controls_no_reference_path():
    """legacy_null=True must ignore ref_null; False must use it."""
    m = make_model(seed=3)
    x, t, text, tmask, ref = _batch(B=2)
    with torch.no_grad():
        m.ref_null.add_(2.0)  # make the null token obviously non-zero

        m.legacy_null = True
        v_legacy_a = m(x, t, text, text_mask=tmask, reference=None)
        v_legacy_b = m(x, t, text, text_mask=tmask, reference=None)
        assert torch.allclose(v_legacy_a, v_legacy_b)

        m.legacy_null = False
        v_new = m(x, t, text, text_mask=tmask, reference=None)

    # with gates open, the learned null token must reach the output
    assert not torch.allclose(v_legacy_a, v_new, atol=1e-4), (
        "learned null token must affect the no-reference output")

    # and mutating ref_null must change the new path but not the legacy one
    with torch.no_grad():
        m.ref_null.add_(1.0)
        v_new2 = m(x, t, text, text_mask=tmask, reference=None)
    assert not torch.allclose(v_new, v_new2, atol=1e-4)


def test_legacy_checkpoint_loading():
    """A v3 state_dict (no ref_null) loads with legacy_null=True."""
    import copy

    src = MCFlowDiT(cfg())
    sd = {k: v.clone() for k, v in src.state_dict().items()
          if k != "ref_null"}
    m = MCFlowDiT(cfg())
    try:
        m.load_state_dict(sd, strict=True)
        raise AssertionError("strict load must fail for a v3 checkpoint")
    except RuntimeError:
        pass
    missing = [k for k in m.state_dict() if k not in sd]
    assert any("ref_null" in k for k in missing)
    m.load_state_dict(sd, strict=False)
    m.legacy_null = True
    x, t, text, tmask, ref = _batch(B=2)
    with torch.no_grad():
        v = m(x, t, text, text_mask=tmask, reference=None)
    assert torch.isfinite(v).all()
    # ref path is zero-gated in this fresh copy -> legacy None == real ref
    with torch.no_grad():
        v2 = m(x, t, text, text_mask=tmask, reference=ref)
    assert torch.allclose(v, v2, atol=1e-5)


def test_freeze_backbone_trains_ref_null():
    """ref_null must be in the trainable set when freeze_backbone is on."""
    m = MCFlowDiT(cfg(ref_size=128))
    trainable_prefixes = ("text_proj", "text_null", "cross_blocks",
                          "head", "head_norm", "t_embedder",
                          "ref_encoder", "ref_adapters", "ref_cross_blocks",
                          "ref_null")
    names = [n for n, p in m.named_parameters() if n.startswith(trainable_prefixes)]
    assert any(n == "ref_null" for n in names), "ref_null must be trainable"
    assert any(n.startswith("ref_encoder") for n in names)
    assert any(n.startswith("ref_null") for n, _ in m.named_parameters())
