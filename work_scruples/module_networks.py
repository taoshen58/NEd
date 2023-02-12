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
from peach.nn_utils.general import slice_tensor_v2, slice_tensor_combine_v2
from .utils import judgment_distribution_to_score

class FeedForwardModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, *arg, **kwargs):
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output


class SelfAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_keys=None, encoder_values=None, across_attention_prior=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_keys is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_keys))
            value_layer = self.transpose_for_scores(self.value(encoder_values))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if across_attention_prior is not None:
            attention_probs = attention_probs * across_attention_prior

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class AttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttentionModule(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_keys=None, encoder_values=None, across_attention_prior=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_keys, encoder_values, across_attention_prior,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class AttentionLayer(nn.Module):
    def __init__(self,
                 config,
                 disable_cross_attention=False,
                 disable_self_attention=False,
                 ):
        super().__init__()
        if not disable_self_attention:
            self.attention = AttentionModule(config)
        if config.add_cross_attention and (not disable_cross_attention):
            self.crossattention = AttentionModule(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def get_extended_attention_mask(self, attention_mask, dtype):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_keys=None, encoder_values=None, across_attention_prior=None,
            encoder_attention_mask=None,
            # past_key_value=None,
            output_attentions=False,   # useless
            *args, **kwargs,
    ):
        seq_len = hidden_states.shape[-2]
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        if hasattr(self, "attention"):
        # if not hasattr(self, "crossattention"):
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=False,
                # past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]
            # # if decoder, the last output is tuple of self-attn cache
            # if self.is_decoder:
            #     outputs = self_attention_outputs[1:-1]
            #     present_key_value = self_attention_outputs[-1]
            # else:
            #     outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        else:
            attention_output = hidden_states
        outputs = tuple()


        # cross_attn_present_key_value = None
        if encoder_keys is not None and hasattr(self, "crossattention"):
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            self_attention_output_checkpoint = attention_output
            # prepare if there need a slice_tensor
            need_split_span = "span_sents" in kwargs
            if need_split_span:
                span_hidden_states, span_attention_mask = slice_tensor_v2(  # [bs,nc,dsl,hn] & [bs,nc,dsl]
                    attention_output, kwargs["span_sents"])
                bs,nc,dsl,hn = span_hidden_states.shape
                attention_output = span_hidden_states.view(-1, dsl, hn)
                span_attention_mask = span_attention_mask.view(-1, dsl)
                span_extended_attention_mask = self.get_extended_attention_mask(
                    span_attention_mask, dtype=span_hidden_states.dtype)
            else:
                bs, nc, dsl, hn = None,None,None,None

            assert attention_output.shape[0] == encoder_keys.shape[0]

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask if not need_split_span else span_extended_attention_mask,
                head_mask,
                encoder_keys, encoder_values, across_attention_prior,
                encoder_attention_mask,
                # cross_attn_past_key_value,
                output_attentions=True,
            )
            attention_output = cross_attention_outputs[0]

            if need_split_span: # recover
                attention_output = attention_output.view(bs,nc,dsl,hn)
                attention_output = slice_tensor_combine_v2(attention_output, kwargs["span_sents"],seq_len)  # [bs,sl,hn]
                # recover cls and sep token
                seq_rgs = torch.arange(  # bs,sl
                    seq_len, dtype=torch.long, device=attention_output.device).unsqueeze(0).expand([bs,-1])
                recover_mask = torch.logical_or(
                    torch.eq(seq_rgs, 0),
                    torch.eq(seq_rgs, (kwargs["seq_lens"] - 1).unsqueeze(-1)),)
                attention_output = torch.where(
                    recover_mask.unsqueeze(-1),
                    self_attention_output_checkpoint,
                    attention_output,
                )

            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

            # # add cross-attn cache to positions 3,4 of present_key_value tuple
            # cross_attn_present_key_value = cross_attention_outputs[-1]
            # present_key_value = present_key_value + cross_attn_present_key_value

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )
        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # # if decoder, return the attn key/values as the last output
        # if self.is_decoder:
        #     outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# =================================
class SupportModule(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.attn_pool = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)
        self.rel_mlp = TwoLayerMLP(
            4 * model_config.hidden_size, 3, model_config.hidden_act,
            middle_dim=model_config.hidden_size, drop_prob=model_config.hidden_dropout_prob
        )

    def forward(self,
                context_hidden_states,  # [bs,nc,sl,hn]
                context_attention_mask,  # [bs,nc,sl]
                support_hidden_states,   # [bs,nc,ssn,ssl,hn]
                support_attention_mask,  # [bs,nc,ssn,ssl]
                *arg, **kwargs):
        # pool_context = self.attn_pool(context_hidden_states, context_attention_mask)  # [bs,nc,hn]
        # pool_support = self.attn_pool(support_hidden_states, support_attention_mask)  # [bs,nc,ssn,hn]

        pool_context = masked_pool(context_hidden_states, context_attention_mask, high_rank=True)
        pool_support = masked_pool(support_hidden_states, support_attention_mask, high_rank=True)

        pool_context = pool_context.unsqueeze(-2).expand_as(pool_support)  # [bs,nc,ssn,hn]

        rel_repr = torch.cat([
            pool_context, pool_support,
            pool_context - pool_support,
            pool_context * pool_support,
        ], dim=-1)

        rel_logits = self.rel_mlp(rel_repr)  # [bs,nc,ssn,3] 0 for contradiction, 1 for neutral, 2 for entailment
        rel_probs = torch.softmax(rel_logits, dim=-1)  # [bs,nc,ssn,3]
        return rel_logits, rel_probs


class IntegrationModule(nn.Module):
    def __init__(self, model_config, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.config = deepcopy(model_config)
        self.config.add_cross_attention = True
        self.config.is_decoder = False

        if num_layers > 0:  
            self.layer = nn.ModuleList([
                AttentionLayer(self.config, disable_cross_attention=False if idx < num_layers-1 else True)
                for idx in range(num_layers)])
        else:
            self.layer = nn.ModuleList([
                AttentionLayer(self.config, disable_self_attention=True),
                AttentionLayer(self.config, disable_cross_attention=True),
            ])
        """
        self.layer = nn.ModuleList([
            AttentionLayer(self.config, disable_cross_attention=False if idx < num_layers-1 else True)
            for idx in range(num_layers)])
        """

        # judgment embedding: num is 5,
        self.judge_emb_mat = nn.Embedding(num_embeddings=5, embedding_dim=self.config.hidden_size)  # [5,hn]


    def get_extended_attention_mask(self, attention_mask, input_shape, device, dtype):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask, dtype):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility

        if dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask

    def forward(
            self,
            ctx_hidden_states,  # [bs,nc,sl,hn]
            ctx_attention_mask,  # [bs,nc,sl]
            sup_hidden_states,  # [bs,nc,ssn,ssl,hn]
            sup_attention_mask,  # [bs,nc,ssn,ssl,hn]
            sup_judgment_probs,  # [bs,nc,ssn,5]
            across_attention_prior,  # [bs,nc,ssn,ssl]
            output_attentions=None,  # useless
            *arg, **kwargs):
        if "span_sents" in kwargs:
            bs, sl, hn = ctx_hidden_states.shape
            nc = None
        else:
            bs, nc, sl, hn = ctx_hidden_states.shape
        _, _, ssn, ssl, _ = sup_hidden_states.shape

        judge_emb = torch.matmul(sup_judgment_probs, self.judge_emb_mat.weight)  # [bs,nc,ssn,hn]
        judge_emb = judge_emb.unsqueeze(-2).expand_as(sup_hidden_states)  # [bs,nc,ssn,ssl,hn]

        all_self_attentions = None
        all_cross_attentions = tuple()

        hidden_states = ctx_hidden_states.view(-1, sl, hn)   # here, no nc
        attention_mask = ctx_attention_mask.view(-1, sl)
        encoder_keys = sup_hidden_states.view(-1, ssn*ssl, hn)
        encoder_values = (sup_hidden_states + judge_emb).view(-1, ssn*ssl, hn)
        across_attention_prior_rsp = across_attention_prior.view(-1, ssn*ssl)[:, None, None, :]
        encoder_attention_mask = sup_attention_mask.view(-1, ssn*ssl)

        seq_lens = attention_mask.sum(dim=-1)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, hidden_states.shape[:-1], hidden_states.device, hidden_states.dtype)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask, hidden_states.dtype)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = None
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
                layer_head_mask,
                encoder_keys, encoder_values, across_attention_prior_rsp,
                encoder_extended_attention_mask,
                # past_key_value,
                output_attentions,  # useless
                seq_lens=seq_lens,
                *arg, **kwargs,
            )
            hidden_states = layer_outputs[0]

            if hasattr(layer_module, "crossattention"):
                all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # pooled_output = pooled_output.view(bs, nc, hn)
        if "span_sents" in kwargs:
            hidden_states_output = hidden_states.view(bs, sl, hn)
        else:
            hidden_states_output = hidden_states.view(bs, nc, sl, hn)

        output_dict = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states_output,
            past_key_values=None,
            hidden_states=None,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        # output_dict["pooled_output"] = pooled_output  # [bs,nc,hn]

        return output_dict


class JudgmentModule(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        # for distill
        num_labels = 2
        # main
        self.pooling = AttentionPooler(
            model_config.hidden_size, model_config.hidden_act, model_config.hidden_dropout_prob)
        self.ffn = FeedForwardModule(model_config)

        self.cls_head = SimpleClassificationHead(model_config.hidden_size, num_labels)

    def convert_probs_to_scores(self, probs):
        # return judgment_distribution_to_score(probs)
        return probs[..., 1]

    def distill_forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.detach()
        pooled_output = masked_pool(hidden_states, attention_mask, high_rank=True)
        # pooled_output = self.distill_ffn(pooled_output)

        pre_logits = self.ffn(pooled_output)
        logits = self.cls_head(pre_logits)
        probs = torch.softmax(logits, dim=-1)
        # to wrong score
        scores = self.convert_probs_to_scores(probs)  # [0,1] and higher denotes more wrong
        return logits, probs, scores

    def forward(self, hidden_states, attention_mask, *arg, **kwargs):
        pooled_output = self.pooling(hidden_states, attention_mask)
        pre_logits = self.ffn(pooled_output)
        logits = self.cls_head(pre_logits)
        probs = torch.softmax(logits, dim=-1)
        # to wrong score
        scores = self.convert_probs_to_scores(probs)  # [0,1] and higher denotes more wrong

        return logits, probs, scores


