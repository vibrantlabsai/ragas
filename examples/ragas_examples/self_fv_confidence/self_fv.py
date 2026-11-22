"""
Self-Function-Vector (self-FV) core — native-PyTorch-hook reimplementation.

Reimplements the mechanic of Chung et al. 2026 (arXiv:2606.19353),
"Quantifying Aleatoric Uncertainty of In-Context Learning", generalized to
modern GQA architectures (Qwen3, Gemma-4) where head_dim is DECOUPLED from
hidden_size / num_attention_heads.

No baukit dependency: extraction uses a forward-pre-hook on each o_proj module
(its input == per-head attention output, concatenated), injection uses a
forward-hook on a decoder layer that adds a vector to the last-token residual.

Pipeline (per Section 3 of the paper):
  S2  self-FV construction : v = sum over top heads of  o_proj_layer( head_slot(a_head) )
  S3  intervention         : h'_{l_t} = h_{l_t} + v   at the final token
  S4  uncertainty decomp   : Total = H[p(y|x)],  AU = H[p(y|x, v)],  EU = Total - AU
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Architecture-generic accessors
# ----------------------------------------------------------------------------
def load_lm(model_id, dtype=None):
    """Load a causal LM, falling back to multimodal (image-text-to-text)
    wrappers like Gemma4ForConditionalGeneration. Returns (tokenizer, model)."""
    import torch as _t
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if dtype is None:
        dtype = _t.bfloat16
    tok = AutoTokenizer.from_pretrained(model_id)
    # device_map="auto" shards large models across all visible GPUs (4xL4);
    # for small models it still fits on one device.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto")
    except (ValueError, KeyError, EnvironmentError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map="auto")
    return tok, model.eval()


def get_text_config(model):
    """Return the text/decoder sub-config (handles multimodal wrappers)."""
    c = model.config
    return getattr(c, "text_config", c)


def get_decoder_layers(model):
    """Return the ModuleList of decoder layers across architectures.

    Qwen3ForCausalLM        -> model.model.layers
    Gemma4ForConditionalGeneration -> model.model.language_model.layers
                                     (or model.language_model.model.layers)
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.text_model.layers,
    ]
    for f in candidates:
        try:
            layers = f(model)
            if layers is not None and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError(f"Could not locate decoder layers for {type(model).__name__}")


def head_dims(model):
    """Return (n_heads, head_dim, hidden), derived from the ACTUAL o_proj input
    dimension rather than config.num_attention_heads — some models (e.g. the
    Gemma-4 E-series) run with a query-head count that differs from config."""
    tc = get_text_config(model)
    hidden = tc.hidden_size
    head_dim = getattr(tc, "head_dim", None) or (hidden // tc.num_attention_heads)
    layers = get_decoder_layers(model)
    first_att = next((l for l in layers if _has_std_attention(l, head_dim)), layers[0])
    o_in = first_att.self_attn.o_proj.in_features
    n_heads = o_in // head_dim
    return n_heads, head_dim, hidden


def _head_dim_of(model):
    tc = get_text_config(model)
    return getattr(tc, "head_dim", None) or (tc.hidden_size // tc.num_attention_heads)


def _has_std_attention(layer, head_dim):
    """True if the layer has a standard nn.Linear self_attn.o_proj whose input
    splits evenly into heads. Excludes linear-attention blocks (Gated DeltaNet)."""
    import torch.nn as nn
    sa = getattr(layer, "self_attn", None)
    op = getattr(sa, "o_proj", None) if sa is not None else None
    return isinstance(op, nn.Linear) and op.in_features % head_dim == 0


def attention_layer_indices(model):
    """Decoder-layer indices that use standard softmax attention (have o_proj).

    On mechanism-hybrid models (e.g. Qwen3.6-35B-A3B: 30/40 layers are Gated
    DeltaNet linear attention) this returns only the standard-attention layers.
    On Gemma-4 / Qwen3 every layer qualifies."""
    hd = _head_dim_of(model)
    return [i for i, l in enumerate(get_decoder_layers(model))
            if _has_std_attention(l, hd)]


def layer_head_count(model, layer_idx):
    """Query-head count for one attention layer = o_proj.in_features // head_dim."""
    hd = _head_dim_of(model)
    return get_decoder_layers(model)[layer_idx].self_attn.o_proj.in_features // hd


# ----------------------------------------------------------------------------
# Extraction: capture per-head o_proj INPUT at the final token
# ----------------------------------------------------------------------------
class HeadActExtractor:
    """Registers forward-pre-hooks on every layer's o_proj to grab its input.

    o_proj input has shape [batch, seq, n_heads * head_dim]; we keep the last
    token and reshape to [n_heads, head_dim]. Stored per layer index.
    """

    def __init__(self, model):
        self.model = model
        self.layers = get_decoder_layers(model)
        self.att_idxs = attention_layer_indices(model)
        self.n_heads, self.head_dim, self.hidden = head_dims(model)
        self._store = {}
        self._handles = []

    def _make_hook(self, layer_idx):
        def pre_hook(module, args, kwargs=None):
            x = args[0]  # [batch, seq, n_heads_layer*head_dim]
            last = x[:, -1, :]
            nh = last.shape[-1] // self.head_dim  # per-layer head count
            self._store[layer_idx] = last.detach().reshape(
                last.shape[0], nh, self.head_dim
            )
        return pre_hook

    def __enter__(self):
        self._store = {}
        for i in self.att_idxs:  # only standard-attention layers
            h = self.layers[i].self_attn.o_proj.register_forward_pre_hook(
                self._make_hook(i), with_kwargs=True
            )
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def head_activation(self, layer_idx, head_idx):
        """[batch, head_dim] activation of one head at the final token."""
        return self._store[layer_idx][:, head_idx, :]


def build_self_fv(extractor: HeadActExtractor, top_heads, model):
    """S2: construct the self-function-vector from the current prompt's own
    head activations, projected through each head's layer o_proj.

    top_heads: list of (layer_idx, head_idx).
    Returns a [hidden] tensor on the model's device/dtype.
    """
    head_dim = _head_dim_of(model)
    layers = get_decoder_layers(model)
    fv = None
    acc_dev = None
    for (layer_idx, head_idx) in top_heads:
        o_proj = layers[layer_idx].self_attn.o_proj
        a = extractor.head_activation(layer_idx, head_idx)  # [batch, head_dim]
        in_features = o_proj.weight.shape[1]  # per-layer o_proj input dim
        x = torch.zeros(a.shape[0], in_features,
                        device=o_proj.weight.device, dtype=o_proj.weight.dtype)
        x[:, head_idx * head_dim:(head_idx + 1) * head_dim] = a.to(x.device, x.dtype)
        contrib = torch.matmul(x, o_proj.weight.T)  # [batch, hidden]
        # layers may live on different GPUs (device_map="auto"); sum on one device
        if fv is None:
            acc_dev = contrib.device
            fv = contrib
        else:
            fv = fv + contrib.to(acc_dev)
    return fv  # [batch, hidden]


# ----------------------------------------------------------------------------
# Injection: add the self-FV to the final-token residual at layer l_t
# ----------------------------------------------------------------------------
class FVInjector:
    def __init__(self, model, target_layer_idx, fv):
        self.layers = get_decoder_layers(model)
        self.target_layer_idx = target_layer_idx
        self.fv = fv
        self._handle = None

    def _hook(self, module, args, output):
        # HF decoder layers return a tuple (hidden_states, ...) or a tensor.
        if isinstance(output, tuple):
            hs = output[0]
            hs[:, -1, :] = hs[:, -1, :] + self.fv.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(output[1:])
        else:
            output[:, -1, :] = output[:, -1, :] + self.fv.to(output.dtype).to(output.device)
            return output

    def __enter__(self):
        layer = self.layers[self.target_layer_idx]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()


# ----------------------------------------------------------------------------
# Uncertainty decomposition over a discrete label space
# ----------------------------------------------------------------------------
def label_entropy(logits_last, option_token_ids):
    """Entropy of p(y | ...) restricted to the option token ids (nats)."""
    cols = torch.tensor(option_token_ids, device=logits_last.device)
    sliced = logits_last.index_select(-1, cols)
    logp = F.log_softmax(sliced, dim=-1)
    return -(logp.exp() * logp).sum(-1)


# ----------------------------------------------------------------------------
# S1: Causal head selection (CIE)  — mean head activation + head replacement
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_mean_head_act(model, list_of_inputs, att_layers=None):
    """Average o_proj-input head activations at the final token over prompts.
    Returns a dict keyed by ATTENTION-layer index: mean_act[li] is
    [n_heads_layer, head_dim]. Non-attention layers (Gated DeltaNet) are absent.
    att_layers restricts to a subset of attention layers (cost control)."""
    att_idxs = att_layers if att_layers is not None else attention_layer_indices(model)
    head_dim = _head_dim_of(model)
    acc = {li: torch.zeros(layer_head_count(model, li), head_dim,
                           device=model.device, dtype=torch.float32)
           for li in att_idxs}
    cnt = 0
    for inp in list_of_inputs:
        with HeadActExtractor(model) as ext:
            model(**inp)
        for li in att_idxs:
            acc[li] += ext._store[li][0].float().to(acc[li].device)
        cnt += 1
    return {li: a / max(cnt, 1) for li, a in acc.items()}


class HeadReplacer:
    """Forward-pre-hook that overwrites ONE head's o_proj input at the last
    token with a replacement vector (used for causal indirect effect)."""

    def __init__(self, model, layer_idx, head_idx, repl_vec):
        self.layer = get_decoder_layers(model)[layer_idx]
        self.head_idx = head_idx
        self.repl = repl_vec  # [head_dim]
        self.n_heads, self.head_dim, _ = head_dims(model)
        self._handle = None

    def _hook(self, module, args, kwargs):
        x = args[0].clone()
        sl = slice(self.head_idx * self.head_dim, (self.head_idx + 1) * self.head_dim)
        x[:, -1, sl] = self.repl.to(x.dtype).to(x.device)
        return (x,) + tuple(args[1:]), kwargs

    def __enter__(self):
        self._handle = self.layer.self_attn.o_proj.register_forward_pre_hook(
            self._hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()


def _true_prob(logits_last, option_token_ids, true_index):
    cols = torch.tensor(option_token_ids, device=logits_last.device)
    p = F.softmax(logits_last.index_select(-1, cols), dim=-1)
    return p[0, true_index].item()


@torch.no_grad()
def compute_cie(model, noised_inputs, true_indices, mean_head_act, option_token_ids,
                progress=None, att_layers=None):
    """Causal Indirect Effect per head: on label-noised prompts, replacing a
    head's activation with the clean mean should restore the true-label prob
    for causal heads. IE[l,h] = mean_x( p_sub(true) - p_clean(true) ).

    Returns (top_heads_sorted, ie_by_pair) where ie_by_pair maps (layer,head)
    to the mean indirect effect. Handles per-layer head counts.
    """
    att_idxs = att_layers if att_layers is not None else attention_layer_indices(model)
    pairs = [(li, hi) for li in att_idxs
             for hi in range(layer_head_count(model, li))]
    ie = {p: 0.0 for p in pairs}
    for n, (inp, ti) in enumerate(zip(noised_inputs, true_indices)):
        clean_logits = model(**inp).logits[:, -1, :]
        p_clean = _true_prob(clean_logits, option_token_ids, ti)
        for (li, hi) in pairs:
            with HeadReplacer(model, li, hi, mean_head_act[li][hi]):
                sub_logits = model(**inp).logits[:, -1, :]
            ie[(li, hi)] += _true_prob(sub_logits, option_token_ids, ti) - p_clean
        if progress:
            progress(n + 1, len(noised_inputs))
    n_ex = max(len(noised_inputs), 1)
    ie = {p: v / n_ex for p, v in ie.items()}
    top = sorted(pairs, key=lambda p: ie[p], reverse=True)
    return top, ie


@torch.no_grad()
def decompose_uncertainty(model, inputs, option_token_ids, top_heads, target_layer_idx,
                          fv_scale=1.0):
    """Return dict(total, aleatoric, epistemic) for one ICL query.

    Total  = H[p(y|x)]                      (clean forward, Eq. 10)
    AU     = H[p(y|x, self-FV)]             (single top-k injection, Eq. 11, N=1)
    EU     = Total - AU

    fv_scale rescales the injected self-FV (diagnostic knob for magnitude).
    """
    # Clean forward + capture head activations in one pass
    with HeadActExtractor(model) as ext:
        clean_logits = model(**inputs).logits[:, -1, :]
        fv = build_self_fv(ext, top_heads, model)  # [batch, hidden]
    total = label_entropy(clean_logits, option_token_ids).mean().item()

    # Intervened forward
    with FVInjector(model, target_layer_idx, fv[0] * fv_scale):
        au_logits = model(**inputs).logits[:, -1, :]
    au = label_entropy(au_logits, option_token_ids).mean().item()

    return {"total": total, "aleatoric": au, "epistemic": total - au}
