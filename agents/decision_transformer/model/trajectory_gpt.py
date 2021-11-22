import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]



def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.prune_heads
        )

        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)

        if self.scale:
            w = w/ (float(v.size(-1)) ** 0.5)

        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            "implments casual mask"
            mask = self.bias[:, :, ns -nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1,3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1))

        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)

        if k:
            return x.permute(0, 2, 3, 1) # (batch, head, head_features, seq_len)
        else:
            return x.permute(0, 2, 1,3) # (batch, head, seq_len, head_feat)


    def forward(self, hidden_states, layer_past =None,
                attention_mask =None,
                head_mask = None,
                encoder_hidden_states = None,
                encoder_attention_mask = None,
                use_cache = False,
                output_attentions = False):

        if encoder_hidden_states is not None:
            assert hasattr(self, "q_attn"), " if class is used as cross attention, the weight have to be identifies"

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask

        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_head(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[-1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = torch.stack(((key.transpose(-2, -1), value)))
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)

        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)

        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()

        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx,n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)

class AdapterMLP(nn.Module):
    def __init__(self, n_state, config):
        super().__init__()

        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()

        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps= config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_state=None, encoder_attention_mask=None,
                use_cache=False, output_attentions=False):
        attn_outputs = self.attn(self.ln_1(hidden_states),
                                 layer_past=layer_past,
                                 attention_mask=attention_mask,
                                 head_mask=head_mask,
                                 use_cache=use_cache,
                                 output_attentions=output_attentions)

        attn_output = attn_outputs[0]
        outputs =attn_outputs[1:]

        hidden_states = attn_output + hidden_states

        if encoder_hidden_state is not None:
            assert hasattr(self, "crossattention"), "if encoder hidden states are passed "
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask = attention_mask,
                head_mask = head_mask,
                encoder_hidden_state = encoder_hidden_state,
                encoder_attention_mask = encoder_attention_mask,
                output_attentions = output_attentions
            )

            attn_output = cross_attn_outputs[0]
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))

        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs

class GPT2PreTrainedModel(PreTrainedModel):
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.zero_()
            module.weight.fill_(1.0)

@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
