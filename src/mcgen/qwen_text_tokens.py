"""Frozen Qwen final hidden states for caption tokens, without global pooling.

The chat template supplies encoder context, but system/role/padding tokens are
not exposed to the generator. Canonical singleton forwards make cached and live
features agree. Raw states are normalized per token in PixelFlow before projection.
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import torch

from .config import fingerprint
from .qwen_embedding import MAX_INPUT_LENGTH, QwenVLEmbedding, conversations

PROTOCOL = 'qwen3_vl_caption_hidden_tokens_v1'


def token_identity(cfg, model_cfg, files):
    parent = fingerprint({'config': {k:v for k,v in asdict(cfg).items() if k != 'cache_mb'},
        'protocol':'qwen3_vl_final_attended_l2_v1', 'max_input_length':MAX_INPUT_LENGTH,
        'output_shape':[1,4096], 'files':files})
    return identity_from_parent(parent, model_cfg.text_length)


def identity_from_parent(parent, max_tokens):
    return fingerprint({'source_encoder':parent, 'protocol':PROTOCOL,
        'max_output_tokens':max_tokens, 'max_input_length':MAX_INPUT_LENGTH,
        'selection':'user-content character offsets; exclude special tokens',
        'features':'last_hidden_state before pooling and L2 normalization',
        'batch_policy':'singleton', 'padding':'right; compact original token order'})


def tokenize_captions(processor, prompts, instruction='', max_tokens=512):
    if not prompts or any(not isinstance(p,str) for p in prompts):
        raise ValueError('Nonempty caption list of strings required')
    sentinel = 'MCGEN_CAPTION_CONTENT_7efb42851'
    template = processor.apply_chat_template(conversations([sentinel], instruction),
        add_generation_prompt=True, tokenize=False)[0]
    if template.count(sentinel) != 1:
        raise ValueError('Cannot identify the caption span in this chat template')
    prefix, suffix = template.split(sentinel)
    rendered = processor.apply_chat_template(conversations(prompts,instruction),
        add_generation_prompt=True, tokenize=False)
    tokens = processor.tokenizer(rendered, add_special_tokens=False, padding=False,
        truncation=False, return_offsets_mapping=True, return_attention_mask=False)
    special = set(processor.tokenizer.all_special_ids)
    result = []
    for prompt, text, ids, offsets in zip(prompts, rendered, tokens['input_ids'], tokens['offset_mapping']):
        if not prompt.strip():
            result.append({'input_ids':[], 'positions':[]})
            continue
        if text != prefix+prompt+suffix:
            raise ValueError('Caption rendering differs from the verified template span')
        if len(ids) > MAX_INPUT_LENGTH:
            raise ValueError('Caption input exceeds512 tokens; truncation is forbidden')
        start, stop = len(prefix), len(prefix)+len(prompt)
        positions = [i for i, (a,b) in enumerate(offsets)
                     if b > a and b > start and a < stop and ids[i] not in special]
        if not positions or len(positions) > max_tokens:
            raise ValueError('Caption tokens are empty or exceed the output token limit')
        result.append({'input_ids':list(ids), 'positions':positions})
    if len(result) != len(prompts):
        raise ValueError('Tokenizer returned a different caption count')
    return result


class QwenVLTextTokens(QwenVLEmbedding):
    def __init__(self,cfg,model_cfg,device):
        if model_cfg.text_dim != 4096 or not 1 <= model_cfg.text_length <= MAX_INPUT_LENGTH:
            raise ValueError('Qwen text tokens require dimension4096 and at most512 tokens')
        super().__init__(cfg,replace(model_cfg,text_length=1),device)
        self.identity = identity_from_parent(self.identity, model_cfg.text_length)
        self.max_tokens = model_cfg.text_length
        self.multimodal_identity = None

    @torch.inference_mode()
    def encode_prepared(self,row,output_device='cpu'):
        positions = row['positions']
        if not positions:
            return torch.empty(0,4096,dtype=getattr(torch,self.cfg.dtype),device=output_device)
        if len(row['input_ids']) > MAX_INPUT_LENGTH or len(positions) > self.max_tokens:
            raise ValueError('Prepared caption exceeds the token protocol')
        ids = torch.tensor([row['input_ids']],dtype=torch.long,device=self.device)
        output = self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        values = output.last_hidden_state[0,positions]
        if values.shape != (len(positions),4096) or not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Invalid caption hidden states')
        self.hits += 1
        return values.to(output_device)

    @torch.inference_mode()
    def encode(self,prompts,output_device):
        rows = tokenize_captions(self.processor,prompts,self.cfg.instruction,self.max_tokens)
        chunks = [self.encode_prepared(row,output_device) for row in rows]
        length = max(1,max(len(h) for h in chunks))
        values = chunks[0].new_zeros(len(chunks),length,4096)
        mask = torch.zeros(len(chunks),length,dtype=torch.bool,device=output_device)
        for i,h in enumerate(chunks):
            values[i,:len(h)] = h
            mask[i,:len(h)] = True
        return values,mask

    def encode_multimodal(self,*args,**kwargs):
        raise ValueError('Token text backend uses a separate visual_tokens reference path')

    def cache_stats(self):
        return {'backend':PROTOCOL,'hits':self.hits,'misses':0}
