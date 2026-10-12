"""Separate ST models and a shared encoder with two independent MTL heads."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

LABEL_COUNTS = {"sentiment": 4, "topic": 10}


class EncoderModel(nn.Module):
    def __init__(self, encoder, tasks, *, single_task=False, mlm=False,
                 head_dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.encoder = encoder
        self.tasks = tuple(tasks)
        self.single_task = single_task
        self.has_mlm = mlm
        if single_task:
            if len(self.tasks) != 1 or mlm:
                raise ValueError("ST requires exactly one task and no MLM")
        else:
            if self.tasks != ("sentiment", "topic"):
                raise ValueError("MTL task order must be sentiment, topic")
            hidden = encoder.config.hidden_size
            self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                        nn.LayerNorm(hidden, eps=layer_norm_eps))
            self.dropout = nn.Dropout(head_dropout)
            self.heads = nn.ModuleDict({t: nn.Linear(hidden, LABEL_COUNTS[t]) for t in tasks})
            if mlm:
                # Randomly initialized, untied head, matching the public three-head design.
                self.mlm_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                             nn.LayerNorm(hidden, eps=layer_norm_eps),
                                             nn.Linear(hidden, encoder.config.vocab_size))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None):
        inputs = {"attention_mask": attention_mask}
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        if inputs_embeds is None:
            inputs["input_ids"] = input_ids
        else:
            inputs["inputs_embeds"] = inputs_embeds
        out = self.encoder(**inputs)
        if self.single_task:
            return {self.tasks[0]: out.logits}
        pooled = self.dropout(self.shared(out.last_hidden_state[:, 0, :]))
        return {task: self.heads[task](pooled) for task in self.tasks}

    def embeddings(self, input_ids):
        return self.encoder.get_input_embeddings()(input_ids)

    def embedding_position_ids(self, input_ids):
        # RoBERTa/XLM-R infer padding-aware positions from IDs; inputs_embeds alone
        # cannot recover them. Preserve the original IDs' positions for SMART.
        if self.encoder.config.model_type in ("roberta", "xlm-roberta"):
            pad = self.encoder.config.pad_token_id
            nonpad = input_ids.ne(pad).long()
            return nonpad.cumsum(dim=1) * nonpad + pad
        return None

    def mlm_logits(self, masked_input_ids, attention_mask):
        if not self.has_mlm:
            raise ValueError("MLM head is disabled")
        encoded = self.encoder(input_ids=masked_input_ids, attention_mask=attention_mask)
        return self.mlm_head(encoded.last_hidden_state)

    def specification(self):
        return {"encoder_config": self.encoder.config.to_dict(), "tasks": list(self.tasks),
                "single_task": self.single_task, "mlm": self.has_mlm,
                "head_dropout": 0.0 if self.single_task else self.dropout.p,
                "layer_norm_eps": 1e-5 if self.single_task else self.shared[2].eps}

    @classmethod
    def from_specification(cls, spec):
        config = dict(spec["encoder_config"])
        kind = config.pop("model_type")
        hf_config = AutoConfig.for_model(kind, **config)
        factory = AutoModelForSequenceClassification if spec["single_task"] else AutoModel
        encoder = factory.from_config(hf_config)
        return cls(encoder, spec["tasks"], single_task=spec["single_task"], mlm=spec["mlm"],
                   head_dropout=spec["head_dropout"], layer_norm_eps=spec["layer_norm_eps"])

    @classmethod
    def from_pretrained(cls, config):
        single = config["variant"].startswith("st_")
        factory = AutoModelForSequenceClassification if single else AutoModel
        kwargs = {"revision": config["model_revision"], "trust_remote_code": False,
                  "output_loading_info": True}
        if single:
            kwargs["num_labels"] = LABEL_COUNTS[config["tasks"][0]]
        encoder, info = factory.from_pretrained(config["model_id"], **kwargs)
        # New ST classifier/pooler weights may be expected. Encoder-body mismatches are fatal.
        body_missing = [k for k in info["missing_keys"]
                        if not any(s in k for s in ("classifier", "score", "pooler"))]
        if body_missing or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"Pretrained encoder loading failed: {info}")
        model = cls(encoder, config["tasks"], single_task=single,
                    mlm=config["mlm"]["enabled"],
                    head_dropout=config["training"]["head_dropout"],
                    layer_norm_eps=config["training"]["head_layer_norm_eps"])
        return model, info
