# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# Ported to PyTorch for PytorchOCR
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch port of PP-FormulaNet Head.
Reference: PaddleOCR/ppocr/modeling/heads/rec_ppformulanet_head.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from collections import OrderedDict
from dataclasses import dataclass, fields

from .unimernet_head import (
    MBartConfig,
    MBartForCausalLM,
    MBartDecoder,
    CustomMBartForCausalLM,
    CustomMBartDecoder,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutputWithCrossAttentionsAndCounting,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    UniMERNetHead,
    ModelOutput,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_export,
    AttentionMaskConverter,
)


# =============================================================================
# PPFormulaNet Head
# =============================================================================

class PPFormulaNet_Head(UniMERNetHead):
    """PPFormulaNet Head with MBart-style decoder.

    Args:
        max_new_tokens: Maximum number of new tokens to generate. Default 1536.
        decoder_start_token_id: Start token ID. Default 0.
        temperature: Sampling temperature. Default 0.2.
        do_sample: Whether to sample. Default False.
        top_p: Top-p sampling. Default 0.95.
        in_channels: Input channels. Default 1024.
        decoder_layers: Number of decoder layers. Default 8.
        encoder_hidden_size: Encoder hidden size. Default 1024.
        decoder_ffn_dim: Decoder FFN dimension. Default 4096.
        decoder_hidden_size: Decoder hidden size. Default 1024.
        is_export: Export flag. Default False.
        length_aware: Length-aware mechanism. Default True.
        use_parallel: Parallel decoding. Default False.
        parallel_step: Parallel step count. Default 3.
    """

    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        decoder_layers=8,
        encoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_hidden_size=1024,
        is_export=False,
        length_aware=True,
        use_parallel=False,
        parallel_step=3,
    ):
        # Skip UniMERNetHead.__init__, build from scratch
        nn.Module.__init__(self)

        mbart_config_dict = {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "add_cross_attention": True,
            "add_final_layer_norm": True,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "d_model": decoder_hidden_size,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": decoder_ffn_dim,
            "decoder_layerdrop": 0.0,
            "decoder_layers": decoder_layers,
            "dropout": 0.1,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "eos_token_id": 2,
            "forced_eos_token_id": 2,
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "output_hidden_states": False,
            "max_position_embeddings": (
                max_new_tokens + parallel_step if use_parallel else max_new_tokens
            ),
            "model_type": "mbart",
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "scale_embedding": True,
            "tie_word_embeddings": False,
            "transformers_version": "4.40.0",
            "use_cache": True,
            "use_return_dict": True,
            "vocab_size": 50000,
            "_attn_implementation": "eager",
            "hidden_size": decoder_hidden_size,
            "use_parallel": use_parallel,
            "parallel_step": int(parallel_step),
            "is_export": is_export,
        }
        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.is_export = is_export
        self.max_seq_len = max_new_tokens
        self.config_decoder = MBartConfig(**mbart_config_dict)
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder = CustomMBartForCausalLM(
            self.config_decoder, length_aware=length_aware
        )
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_hidden_size, self.config_decoder.hidden_size
            )
        generation_config = {
            "max_length": 1537,
            "forced_eos_token_id": 2,
        }
        self.eos_token_id = generation_config["forced_eos_token_id"]
        self.pad_token_id = self.config_decoder.pad_token_id
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config["max_length"],
                generation_config["forced_eos_token_id"],
            )
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        decoder_inputs = self.prepare_inputs_for_generation_mbart(
            input_ids, past_key_values=past_key_values
        )
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def _extract_past_from_model_output(
        self, outputs, standardize_cache_format: bool = False
    ):
        past_key_values = None
        if hasattr(outputs, 'past_key_values') and 'past_key_values' in outputs:
            past_key_values = outputs.past_key_values
        elif hasattr(outputs, 'mems') and 'mems' in outputs:
            past_key_values = outputs.mems
        elif hasattr(outputs, 'past_buckets_states') and 'past_buckets_states' in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if hasattr(outputs, "state") and getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def stopping_criteria(self, input_ids):
        if self.is_export:
            return input_ids[:, -1] == self.eos_token_id
        is_done = torch.isin(input_ids[:, -1], torch.tensor([self.eos_token_id], device=input_ids.device))
        return is_done

    def stopping_criteria_parallel(self, input_ids):
        parallel_step = self.config_decoder.parallel_step
        if self.is_export:
            is_done_list = []
            for i in range(parallel_step, 0, -1):
                cur_is_done = input_ids[:, -i] == self.eos_token_id
                is_done_list.append(cur_is_done)
            is_done_list = torch.stack(is_done_list, dim=1)
            return is_done_list
        else:
            is_done = torch.isin(
                input_ids[:, -parallel_step:],
                torch.tensor([self.eos_token_id], device=input_ids.device).reshape([1, 1]),
            )
            return is_done

    def generate_single_iter(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_hidden_states = encoder_outputs[0]
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        kwargs_decoder = {}

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        return Seq2SeqLMOutput(
            loss=None,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, 'hidden_states') else None,
            encoder_attentions=encoder_outputs.attentions if hasattr(encoder_outputs, 'attentions') else None,
        )

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_kwargs,
        decoder_start_token_id=None,
        bos_token_id=None,
    ):
        # 1. Check whether user has defined decoder_input_ids manually
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Get start token
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )

        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(
                decoder_start_token_id, dtype=torch.int64
            )
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            use_parallel = self.config_decoder.use_parallel
            parallel_step = self.config_decoder.parallel_step

            if use_parallel:
                decoder_input_ids_start = (
                    torch.ones((batch_size, parallel_step), dtype=torch.int64)
                    * decoder_start_token_id
                )
            else:
                decoder_input_ids_start = (
                    torch.ones((batch_size, 1), dtype=torch.int64)
                    * decoder_start_token_id
                )

        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (
            self.config_decoder.model_type == "vision-encoder-decoder"
            and "donut" in getattr(self, 'name_or_path', '').lower()
        ):
            pass
        elif self.config_decoder.model_type in ["whisper"]:
            pass
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat(
                [decoder_input_ids_start, decoder_input_ids], dim=-1
            )
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (
                        torch.ones_like(decoder_attention_mask)[:, :parallel_step if use_parallel else 1],
                        decoder_attention_mask,
                    ),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @torch.no_grad()
    def generate(self, model_kwargs):
        use_parallel = self.config_decoder.use_parallel
        parallel_step = self.config_decoder.parallel_step
        batch_size = model_kwargs["encoder_outputs"]["last_hidden_state"].shape[0]
        generation_config = {
            "decoder_start_token_id": 0,
            "bos_token_id": 0,
        }

        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )

        decoder_input_ids = input_ids
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]

        model_kwargs["cache_position"] = torch.arange(cur_len)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id

        if use_parallel:
            unfinished_sequences = torch.ones(
                [batch_size, parallel_step], dtype=torch.int64, device=input_ids.device
            )
            parallel_length = math.ceil(self.max_seq_len // parallel_step)
        else:
            unfinished_sequences = torch.ones(
                batch_size, dtype=torch.int64, device=input_ids.device
            )
            parallel_length = self.max_seq_len

        past_key_values = []

        for idx in range(parallel_length):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.generate_single_iter(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if use_parallel:
                next_token_logits = outputs.logits[:, :, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            if use_parallel:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config_decoder.is_encoder_decoder,
            )

            if use_parallel:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria_parallel(input_ids).to(torch.int64)
                )
            else:
                unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                    input_ids
                ).to(torch.int64)

            if (
                eos_token is not None
                and (
                    torch.cumsum((input_ids == eos_token).to(torch.int64), 1)[:, -1]
                    >= 1
                ).all()
            ):
                break

        return input_ids

    def forwad_train(
        self,
        encoder_outputs,
        decoder_input_ids,
        decoder_attention_mask,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if self.config_decoder.use_parallel:
            batch = decoder_input_ids.shape[0]
            add_sos_token = self.config_decoder.parallel_step - 1
            start_token = torch.zeros([batch, add_sos_token], dtype=torch.int64, device=decoder_input_ids.device)
            start_mask = torch.ones([batch, add_sos_token], dtype=torch.int64, device=decoder_input_ids.device)
            decoder_input_ids = torch.cat([start_token, decoder_input_ids], dim=1)
            decoder_attention_mask = torch.cat(
                [start_mask, decoder_attention_mask], dim=1
            )

        labels = decoder_input_ids * 1
        labels = labels.masked_fill_(labels == self.pad_token_id, -100)
        if self.config_decoder.use_parallel:
            input_decoder_input_ids = decoder_input_ids[
                :, : -self.config_decoder.parallel_step
            ]
            input_decoder_attention_mask = decoder_attention_mask[
                :, : -self.config_decoder.parallel_step
            ]
        else:
            input_decoder_input_ids = decoder_input_ids[:, :-1]
            input_decoder_attention_mask = decoder_attention_mask[:, :-1]

        encoder_hidden_states = encoder_outputs[0]
        kwargs_decoder = {}
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        decoder_outputs = self.decoder(
            input_ids=input_decoder_input_ids,
            attention_mask=input_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits
        return logits, labels

    def forward(self, inputs, targets=None):
        self.is_export = False if self.training else True
        if not self.training:
            encoder_outputs = inputs
            model_kwargs = {
                "output_attentions": False,
                "output_hidden_states": False,
                "use_cache": True,
                "encoder_outputs": encoder_outputs,
            }
            word_pred = self.generate(model_kwargs)
            return word_pred

        encoder_outputs, tgt_seq, mask = inputs
        logits, masked_labels = self.forwad_train(encoder_outputs, tgt_seq, mask)
        return logits, masked_labels
