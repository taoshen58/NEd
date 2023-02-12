import math
from collections import OrderedDict

import torch

from transformers.models.auto.modeling_auto import auto_class_factory

from transformers.models.roberta.modeling_roberta import *

from peach.nn_utils.nn import TwoLayerMLP
from peach.nn_utils.general import masked_pool, zero_mask, exp_mask

from .module_networks import *
from .utils import *



class RobertaForDilemmas(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.support_scorer = SupportModule(model_config=config)
        self.integration_module = IntegrationModule(model_config=config, num_layers=2)
        self.judgment_scorer = JudgmentModule(model_config=config)


    def deep_encoding(self,
                 input_ids,
                 attention_mask,
                 position_ids=None,
    ):
        input_shape = input_ids.shape
        sl = input_shape[-1]

        outputs = self.roberta(
            input_ids.view(-1, sl),
            attention_mask=attention_mask.view(-1, sl),
            position_ids=position_ids.view(-1, sl) if position_ids is not None else None,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # extract hidden states
        hidden_states = outputs[0]
        hn = hidden_states.shape[-1]
        new_shape = list(input_shape) + [hn,]
        return hidden_states.view(*new_shape)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            # head_mask=None,
            # inputs_embeds=None,
            labels=None,
            # output_attentions=None,
            # output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        # extra_outputs = dict()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs["sup_wrong_scores"] = judgment_distribution_to_score(kwargs["sup_judgment_probs"])
        kwargs["sup_wrong_probs"] = torch.stack(
            [1. - kwargs["sup_wrong_scores"], kwargs["sup_wrong_scores"]], dim=-1).contiguous()

        extra_outputs = {
            "sup_wrong_probs": kwargs["sup_wrong_probs"],
        }

        sup_input_ids = kwargs["sup_input_ids"]
        sup_attention_mask = kwargs["sup_attention_mask"]
        sup_sent_mask = (sup_attention_mask.sum(dim=-1) > 0).to(torch.long)

        bs, nc, sl = input_ids.shape
        bs, nc, ssn, ssl = kwargs["sup_input_ids"].shape

        # deep contextualized embedding
        ctx_hidden_states = self.deep_encoding(  # [bs, nc, sl, hn]
            input_ids, attention_mask, position_ids)

        if self.config.enable_support:
            sup_hidden_states = self.deep_encoding(  # [bs, nc, ssn, ssl, hn]
                sup_input_ids, sup_attention_mask, kwargs.get("sup_position_ids"),)

        # check support scores
        if self.config.rel_prior_source == "learned":
            assert self.config.enable_support
            rel_logits, rel_probs = self.support_scorer(  # [bs,nc,ssn,3]
                ctx_hidden_states, attention_mask,
                sup_hidden_states, sup_attention_mask,
            )
            if self.training:
                distill_rel_logits, distill_rel_probs = self.support_scorer(
                    ctx_hidden_states.detach(), attention_mask,
                    sup_hidden_states.detach(), sup_attention_mask,
                )
            else:
                distill_rel_logits, distill_rel_probs = rel_logits, rel_probs
            extra_outputs["distill_rel_logits"] = distill_rel_logits

            rel_prior = (1. - rel_probs[..., 1])  # [bs,nc,ssn]
            rel_prior = rel_prior.unsqueeze(-1).expand_as(sup_attention_mask).contiguous()  # [bs,nc,ssn,ssl]
        elif self.config.rel_prior_source == "nli":
            rel_prior = (1. - kwargs["sup_nli_probs"][..., 1])  # [bs,nc,ssn]
            rel_prior = rel_prior.unsqueeze(-1).expand_as(sup_attention_mask).contiguous()  # [bs,nc,ssn,ssl]
        elif self.config.rel_prior_source == "ones":
            rel_prior = torch.ones([bs, nc, ssn, ssl], dtype=ctx_hidden_states.dtype, device=ctx_hidden_states.device)
        else:
            raise NotImplementedError(self.config.rel_prior_source)


        # integration
        # 1. distill for self.judgment_scorer
        if self.config.enable_support:
            distill_sup_wrong_logits, distill_sup_wrong_probs, distill_sup_wrong_scores = \
                self.judgment_scorer.distill_forward(sup_hidden_states, sup_attention_mask)
            extra_outputs["distill_sup_wrong_logits"] = distill_sup_wrong_logits

        # 2.
        if self.config.enable_integration:
            assert self.config.enable_support
            integration_outputs = self.integration_module(
                ctx_hidden_states, attention_mask,
                sup_hidden_states, sup_attention_mask, kwargs["sup_judgment_probs"],
                across_attention_prior=rel_prior,
            )
            # visualization
            cross_attentions = integration_outputs[1][0].view(bs,nc,-1,sl,ssn,ssl)
            extra_outputs["explain_cross_attentions"] = cross_attentions.mean(2).sum(-1).max(2)[0]  # [bs,nc,sl,ssn] -> [bs,nc,ssn]
            extra_outputs["explain_rel_priors"] = rel_prior.mean(-1) # [bs,nc,ssn]
            integration_hidden_states = integration_outputs["last_hidden_state"]  # [bs,nc,sl,hn]
        else:
            integration_hidden_states = ctx_hidden_states

        judge_logits, judge_probs, judge_scores = self.judgment_scorer(  # [bs,nc,2] & [bs,nc,2] & [bs,nc]
            integration_hidden_states, attention_mask)
        extra_outputs["judge_logits"], extra_outputs["judge_probs"], extra_outputs["judge_scores"] = judge_logits, judge_probs, judge_scores
        logits = judge_scores.log()

        loss = None
        if labels is not None:
            loss = 0.

            extra_outputs["main_loss"] = calculate_binary_logits_loss(
                logits, labels, "thresh_soft_label", **kwargs)
            loss += extra_outputs["main_loss"]

            # Distillation
            gamma = 1. if self.config.enable_distillation else 0.
            anneal_factor = self.anneal_axu_loss(gain=10.0, **kwargs)
            # loss for rel
            if "distill_rel_logits" in extra_outputs:
                extra_outputs["nli_distill_loss"] = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(extra_outputs["distill_rel_logits"].view(-1, 3)[sup_sent_mask.view(-1) == 1], dim=-1),
                    kwargs["sup_nli_probs"].view(-1, 3)[sup_sent_mask.view(-1) == 1],
                )
                loss += gamma * anneal_factor * extra_outputs["nli_distill_loss"]

            # loss for judgment
            if "distill_sup_wrong_logits" in extra_outputs:
                extra_outputs["judgment_distill_loss"] = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(extra_outputs["distill_sup_wrong_logits"].view(-1, 2)[sup_sent_mask.view(-1) == 1], dim=-1),
                    kwargs["sup_wrong_probs"].view(-1, 2)[sup_sent_mask.view(-1) == 1],
                )
                loss += gamma * anneal_factor * extra_outputs["judgment_distill_loss"]

        main_output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

        for k in extra_outputs:
            main_output[k] = extra_outputs[k]

        return main_output

    def anneal_axu_loss(self, gain=1., **kwargs):
        if "training_progress" not in kwargs:
            return 1.
        training_progress = kwargs["training_progress"]
        assert 0. <= training_progress <= 1., training_progress

        return math.exp(-training_progress * gain)


MODEL_FOR_DILEMMAS_MAPPING = OrderedDict(
    [
        # (BertConfig, ),
        (RobertaConfig, RobertaForDilemmas),

    ]
)

AutoModelForDilemmas = auto_class_factory(
    "AutoModelForDilemmas",
    MODEL_FOR_DILEMMAS_MAPPING,
    head_doc="bool sequence classification for Dilemmas"
)

