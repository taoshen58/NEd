import math
from collections import OrderedDict

import torch

from transformers.models.auto.modeling_auto import auto_class_factory

from transformers.models.roberta.modeling_roberta import *

from peach.nn_utils.nn import TwoLayerMLP
from peach.nn_utils.general import masked_pool, zero_mask, exp_mask, slice_tensor_v2, slice_tensor_combine_v2

from .utils import *

import torch, math
from torch import nn
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate, RobertaOutput,
    RobertaSelfOutput,
    ACT2FN, BaseModelOutputWithPastAndCrossAttentions,
)
from peach.nn_utils.general import masked_pool, zero_mask, exp_mask
from copy import deepcopy
from peach.nn_utils.nn import (
    TwoLayerMLP, SimpleClassificationHead, AttentionPooler,
)

from .module_networks import SupportModule, IntegrationModule, JudgmentModule, FeedForwardModule


class DecomposeJudgmentModule(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        num_labels = 5
        self.pooling = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)
        self.cls_head = MyRobertaClassificationHead(model_config, num_labels=num_labels)

        self.distill_ffn = FeedForwardModule(config=model_config)

    def distill_forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.detach()
        pooled_output = self.distill_ffn(masked_pool(hidden_states, attention_mask, high_rank=True))
        org_lgt = self.cls_head(pooled_output)
        cls_logits = torch.stack([org_lgt[..., 0], org_lgt[..., 1] + org_lgt[..., 2] + org_lgt[..., 3]], dim=-1)
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_scores = cls_probs[..., 1]
        return cls_logits, cls_probs, cls_scores

    def forward(self, hidden_states, attention_mask):
        logits = self.cls_head(self.pooling(hidden_states, attention_mask))

        author_logits = torch.stack([logits[..., 0]+logits[..., 2], logits[..., 1]+logits[..., 3]], dim=-1)
        author_probs = torch.softmax(author_logits, dim=-1)
        author_scores = author_probs[..., 1]

        other_logits = torch.stack([logits[..., 0]+logits[..., 1], logits[..., 2]+logits[..., 3]], dim=-1)
        other_probs = torch.softmax(other_logits, dim=-1)
        other_scores = other_probs[..., 1]

        meta_dict = {
            "author_wrong_logits": author_logits,
            "author_wrong_probs": author_probs,
            "author_wrong_scores": author_scores,
            "other_wrong_logits": other_logits,
            "other_wrong_probs": other_probs,
            "other_wrong_scores": other_scores,
        }

        return logits, meta_dict


class SimpleJudgmentModule(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        num_labels = 2
        self.pooling1 = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)
        self.pooling2 = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)

        # self.cls_head = SimpleClassificationHead(model_config.hidden_size, num_labels)
        # self.info_head = SimpleClassificationHead(model_config.hidden_size, 1)
        self.cls_head = MyRobertaClassificationHead(model_config, num_labels=num_labels)
        self.info_head = SimpleClassificationHead(model_config.hidden_size, num_labels=1)

        # for distill
        self.distill_ffn = FeedForwardModule(config=model_config)

    def distill_forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.detach()
        pooled_output = masked_pool(hidden_states, attention_mask, high_rank=True)

        cls_logits = self.cls_head(self.distill_ffn(pooled_output))  # logits
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_scores = cls_probs[..., 1]
        return cls_logits, cls_probs, cls_scores

    def forward(self, hidden_states, attention_mask):
        pooled_output1 = self.pooling1(hidden_states, attention_mask)  # bs,2
        pooled_output2 = self.pooling2(hidden_states, attention_mask)  # bs,2
        pooled_cls = hidden_states[..., 0, :]

        cls_output1 = self.cls_head(pooled_output1)
        cls_output2 = self.cls_head(pooled_output2)
        info_output = self.info_head(pooled_cls)

        pre_logits = (cls_output1.unsqueeze(-2) + cls_output2.unsqueeze(-1)).view(-1, 4)

        logits = torch.cat(  # bs,5
            [pre_logits, info_output], dim=-1
        )

        author_logits = cls_output1
        author_probs = torch.softmax(author_logits, dim=-1)
        author_scores = author_probs[..., 1]

        other_logits = cls_output2
        other_probs = torch.softmax(other_logits, dim=-1)
        other_scores = other_probs[..., 1]

        meta_dict = {
            "author_wrong_logits": author_logits,
            "author_wrong_probs": author_probs,
            "author_wrong_scores": author_scores,
            "other_wrong_logits": other_logits,
            "other_wrong_probs": other_probs,
            "other_wrong_scores": other_scores,
        }

        return logits.contiguous(), meta_dict



class MyRobertaClassificationHead(nn.Module):
    def __init__(self, config, num_labels=None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels or config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ================
use_ensemble_logits = True
enable_support = True

# use_judgment_distill = False
# use_loss_weights = True
# use_decompose = True
# ================

class RobertaForAnecdotes(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.support_scorer = SupportModule(model_config=config)
        self.integration_module = IntegrationModule(model_config=config, num_layers=0)
        # self.judgment_scorer = DualJudgmentModule(model_config=config)

        # self.judgment_scorer1 = JudgmentModule(model_config=config)
        # self.judgment_scorer2 = JudgmentModule(model_config=config)

        self.cls_pool = AttentionPooler(config.hidden_size, config.hidden_act, config.hidden_dropout_prob)
        self.classifier = MyRobertaClassificationHead(config)

        # need more information module
        # self.info_pool = AttentionPooler(config.hidden_size, config.hidden_act, config.hidden_dropout_prob)
        # self.info_mlp = TwoLayerMLP(
        #     8+config.hidden_size, config.hidden_size, config.hidden_act,
        #     config.intermediate_size, config.hidden_dropout_prob)
        # self.info_cls = SimpleClassificationHead(config.hidden_size, 2)

        if self.config.use_decompose:
            self.simple_judgment_scorer = DecomposeJudgmentModule(model_config=config)
        else:
            self.simple_judgment_scorer = SimpleJudgmentModule(model_config=config)

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs["sup_wrong_scores"] = judgment_distribution_to_score(kwargs["sup_judgment_probs"])  # [bs,nc,ssn,ssl]
        kwargs["sup_wrong_probs"] = torch.stack(  # [bs,nc,ssn,ssl,2]
            [1. - kwargs["sup_wrong_scores"], kwargs["sup_wrong_scores"]], dim=-1).contiguous()
        extra_outputs = {
            "sup_wrong_probs": kwargs["sup_wrong_probs"],
        }

        sup_input_ids = kwargs["sup_input_ids"]
        sup_attention_mask = kwargs["sup_attention_mask"]
        span_sents = kwargs["span_sents"]  # bs,nc,2
        sup_sent_mask = (sup_attention_mask.sum(dim=-1) > 0).to(torch.long)

        bs, sl = input_ids.shape
        bs, nc, ssn, ssl = kwargs["sup_input_ids"].shape

        # deep contextualized embedding
        ctx_hidden_states = self.deep_encoding(  # [bs, sl, hn]
            input_ids, attention_mask, position_ids)
        dtype, device = ctx_hidden_states.dtype, ctx_hidden_states.device

        if enable_support:
            sup_hidden_states = self.deep_encoding(  # [bs, nc, ssn, ssl, hn]
                sup_input_ids, sup_attention_mask, kwargs.get("sup_position_ids"), )

        # split text to sentence from [bs,sl,hn] to [bs,nc,dsl,hn]/[bs*nc,dsl,hn]
        # combine back [bs,nc,dsl,hn]/[bs*nc,dsl,hn] -> [bs,sl,hn]

        if enable_support:
            # check support scores
            span_sents = kwargs["span_sents"]
            ctx_span_hidden_states, ctx_span_mask = slice_tensor_v2(  # [bs,nc,dsl,hn] & [bs,nc,dsl]
                ctx_hidden_states, span_sents)
            rel_logits, rel_probs = self.support_scorer(  # [bs,nc,ssn,3]
                ctx_span_hidden_states, ctx_span_mask,
                sup_hidden_states, sup_attention_mask,
            )
            if self.training:
                distill_rel_logits, distill_rel_probs = self.support_scorer(
                    ctx_span_hidden_states, ctx_span_mask,
                    sup_hidden_states.detach(), sup_attention_mask,
                )
            else:
                distill_rel_logits, distill_rel_probs = rel_logits, rel_probs
            extra_outputs["distill_rel_logits"] = distill_rel_logits

            rel_prior = (1. - rel_probs[..., 1])  # [bs,nc,ssn]
            rel_prior = rel_prior.unsqueeze(-1).expand_as(sup_attention_mask).contiguous()  # [bs,nc,ssn,ssl]
        # nli_rel_prior = (1. - kwargs["sup_nli_probs"][..., 1])  # [bs,nc,ssn]
        # nli_rel_prior = nli_rel_prior.unsqueeze(-1).expand_as(sup_attention_mask).contiguous()  # [bs,nc,ssn,ssl]

        # integration
        # 1. distilllation
        if enable_support:
            # 2.
            integration_hidden_states = ctx_hidden_states
            integration_outputs = self.integration_module(
                ctx_hidden_states, attention_mask,
                sup_hidden_states, sup_attention_mask, kwargs["sup_judgment_probs"],
                # across_attention_prior=nli_rel_prior,
                across_attention_prior=rel_prior,
                # across_attention_prior=torch.ones([bs,nc,ssn,ssl], dtype=ctx_hidden_states.dtype, device=ctx_hidden_states.device),
                span_sents=kwargs["span_sents"],
            )
            # visualization
            cross_attentions = integration_outputs[1][0].view(bs, nc, self.config.num_attention_heads, -1, ssn, ssl)
            extra_outputs["explain_cross_attentions"] = cross_attentions.mean(2).sum(-1).max(2)[0]  # [bs,nc,sl,ssn] -> [bs,nc,ssn]
            extra_outputs["explain_rel_priors"] = rel_prior.mean(-1)  # [bs,nc,ssn]
            integration_hidden_states = integration_outputs["last_hidden_state"]  # [bs,sl,hn]
        else:
            integration_hidden_states = ctx_hidden_states

        if use_ensemble_logits and enable_support and self.config.use_judgment_distill:
            distill_sup_wrong_logits, distill_sup_wrong_probs, distill_sup_wrong_scores = \
                self.simple_judgment_scorer.distill_forward(sup_hidden_states, sup_attention_mask)
            extra_outputs["distill_sup_wrong_logits"] = distill_sup_wrong_logits

        if use_ensemble_logits:
            logits, judgement_meta = self.simple_judgment_scorer(
                integration_hidden_states, attention_mask)
            extra_outputs.update(judgement_meta)
        else:
            logits = self.classifier(self.cls_pool(integration_hidden_states, attention_mask), pooling=False)

        loss = None
        if labels is not None:
            loss = 0.

            if use_ensemble_logits:
                if self.config.use_loss_weights:
                    weights = torch.tensor([2.0, 2.0, 1.0, 3.0, 5.], dtype=logits.dtype, device=logits.device)
                else:
                    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.], dtype=logits.dtype, device=logits.device)

                soft_labels = kwargs.get("soft_labels")
                soft_mask1 = soft_labels >= 0.333
                soft_mask2 = torch.arange(0, 5, device=labels.device, dtype=labels.dtype).unsqueeze(0) == \
                             labels.unsqueeze(1)
                soft_mask = torch.logical_or(soft_mask1, soft_mask2)
                # add_weight to mask
                soft_labels = soft_mask.to(soft_labels.dtype) * soft_labels  # [..., nc]
                soft_labels = soft_labels * weights
                norm_fct = nn.LogSoftmax(dim=-1)
                pred_logprob = norm_fct(logits.view(-1, 5))
                extra_outputs["main_loss"] = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
                loss += extra_outputs["main_loss"]

                if "author_wrong_logits" in extra_outputs:
                    extra_outputs["author_loss"] = calculate_binary_logits_loss(
                        extra_outputs["author_wrong_logits"], kwargs["author_labels"],
                        problem_type="thresh_soft_label_weighted",
                        soft_labels=kwargs["author_soft_labels"],
                        weights=torch.tensor([1.,1.], dtype=logits.dtype, device=logits.device)
                    )
                    # loss += extra_outputs["author_loss"]

                if "other_wrong_logits" in extra_outputs:
                    extra_outputs["other_loss"] = calculate_binary_logits_loss(
                        extra_outputs["other_wrong_logits"], kwargs["other_labels"],
                        problem_type="thresh_soft_label_weighted",
                        soft_labels=kwargs["other_soft_labels"],
                        weights=torch.tensor([1., 1.], dtype=logits.dtype, device=logits.device)
                    )
                    # loss += extra_outputs["other_loss"]
            else:
                # # !! overall !!
                weights_list = [
                    [3.1, 2.1, 1., 4., 6.],  # ==>
                    [2.0, 2.0, 1.0, 3.0, 5.],
                    # ==>  'precision_macro': 0.38589370307750104, 'recall_macro': 0.40368204829361565, 'f1_macro': 0.380997486724762,
                    [1.6, 3.2, 1.0, 1.9, 4.],
                    # ==> 'precision_macro': 0.3048300581786372, 'recall_macro': 0.3345582959348323, 'f1_macro': 0.3135100716782364,
                    [1.6, 2.0, 1.0, 2.0, 5.],
                    # ==> 'precision_macro': 0.5032450798018167, 'recall_macro': 0.35152642888480146, 'f1_macro': 0.3511378208910072
                    [2.0, 2.0, 1.0, 3.0, 7.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0,]
                ]
                if self.config.use_loss_weights:
                    weights = torch.tensor(weights_list[1], dtype=logits.dtype, device=logits.device)
                else:
                    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.], dtype=logits.dtype, device=logits.device)
                soft_labels = kwargs.get("soft_labels")
                soft_mask1 = soft_labels >= 0.333
                soft_mask2 = torch.arange(0, 5, device=labels.device, dtype=labels.dtype).unsqueeze(0) == \
                             labels.unsqueeze(1)
                soft_mask = torch.logical_or(soft_mask1, soft_mask2)
                # add_weight to mask
                soft_labels = soft_mask.to(soft_labels.dtype) * soft_labels  # [..., nc]
                soft_labels = soft_labels * weights
                norm_fct = nn.LogSoftmax(dim=-1)
                pred_logprob = norm_fct(logits.view(-1, 5))
                extra_outputs["main_loss"] = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
                loss += extra_outputs["main_loss"]

            # ~ aux
            anneal_factor = self.anneal_axu_loss(gain=10.0, **kwargs)
            # # loss for rel
            if "distill_rel_logits" in extra_outputs:
                extra_outputs["nli_distill_loss"] = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(extra_outputs["distill_rel_logits"].view(-1, 3)[sup_sent_mask.view(-1) == 1], dim=-1),
                    kwargs["sup_nli_probs"].view(-1, 3)[sup_sent_mask.view(-1) == 1],
                )
                loss += anneal_factor * extra_outputs["nli_distill_loss"]
            #
            # # loss for support rel
            if "distill_sup_wrong_logits" in extra_outputs:
                extra_outputs["judgment_distill_loss"] = nn.KLDivLoss(reduction='batchmean')(
                    torch.log_softmax(extra_outputs["distill_sup_wrong_logits"].view(-1, 2)[sup_sent_mask.view(-1) == 1], dim=-1),
                    kwargs["sup_wrong_probs"].view(-1, 2)[sup_sent_mask.view(-1) == 1],
                )
                loss += anneal_factor * extra_outputs["judgment_distill_loss"]

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



MODEL_FOR_ANECDOTES_MAPPING = OrderedDict(
    [
        # (BertConfig, ),
        (RobertaConfig, RobertaForAnecdotes),

    ]
)

AutoModelForAnecdotes = auto_class_factory(
    "AutoModelForAnecdotes",
    MODEL_FOR_ANECDOTES_MAPPING,
    head_doc="bool sequence classification for Anecdotes"
)

class DualJudgmentModule(JudgmentModule):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.pooling2 = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, *arg, **kwargs):
        logits1, probs1, scores1 = super().forward(hidden_states, attention_mask, *arg, **kwargs)

        pooled_output = self.pooling2(hidden_states, attention_mask)
        pre_logits = self.ffn(pooled_output)
        logits = self.cls_head(pre_logits)
        probs = torch.softmax(logits, dim=-1)
        # to wrong score
        scores = self.convert_probs_to_scores(probs)  # [0,1] and higher denotes more wrong

        return (logits1, probs1, scores1), (logits, probs, scores)

def re_assembling_repre(ctx_hidden_states, ctx_attenton_mask, span_sents):
    seq_len = ctx_hidden_states.shape[-2]
    seq_lens = ctx_attenton_mask.sum(-1)

    span_hidden_states, span_attention_mask = slice_tensor_v2(  # [bs,nc,dsl,hn] & [bs,nc,dsl]
        ctx_hidden_states, span_sents)
    bs, nc, dsl, hn = span_hidden_states.shape

    recover = slice_tensor_combine_v2(span_hidden_states, span_sents, seq_len)  # [bs,sl,hn]

    seq_rgs = torch.arange(  # bs,sl
        seq_len, dtype=torch.long, device=recover.device).unsqueeze(0).expand([bs, -1])
    recover_mask = torch.logical_or(
        torch.eq(seq_rgs, 0),
        torch.eq(seq_rgs, (seq_lens - 1).unsqueeze(-1)), )
    attention_output = torch.where(
        recover_mask.unsqueeze(-1),
        ctx_hidden_states,
        recover,
    )
    return attention_output


