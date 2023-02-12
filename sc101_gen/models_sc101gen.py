from transformers.models.auto.modeling_auto import auto_class_factory
from collections import OrderedDict

from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.bart.modeling_bart import *

from peach.nn_utils.general import masked_pool, zero_mask


def get_final_loss(loss_lm, loss_judgment, loss_agree,):
    return loss_judgment + loss_lm


class BartForSC101Gen(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # classifications
        if self.config.value_to_predict == "action":
            self.judgment_scorer = torch.nn.Sequential(  # special for action
                torch.nn.Linear(config.d_model, 2 * config.d_model),
                nn.GELU(),
                torch.nn.Dropout(config.activation_dropout),
                torch.nn.Linear(2 * config.d_model, 5, bias=False),
            )
            self.agency_scorer = torch.nn.Sequential(
                torch.nn.Linear(config.d_model, 2 * config.d_model),
                nn.GELU(),
                torch.nn.Dropout(config.activation_dropout),
                torch.nn.Linear(2 * config.d_model, 2, bias=False),
            )
        else:
            self.judgment_scorer = None
            self.agency_scorer = None

        # self.agree_scorer = torch.nn.Sequential(
        #     torch.nn.Linear(config.d_model, 2 * config.d_model),
        #     nn.GELU(),
        #     torch.nn.Dropout(config.activation_dropout),
        #     torch.nn.Linear(2 * config.d_model, 5, bias=False),
        # )

        self.char_scorer = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, 2 * config.d_model),
            nn.GELU(),
            torch.nn.Dropout(config.activation_dropout),
            torch.nn.Linear(2 * config.d_model, 3, bias=False),
        )

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        # get last step
        other_outputs = {}
        if decoder_attention_mask is not None:  # todo: also add to generative GPT2
            hidden_states = outputs[0]  # last hidden state
            # if decoder_attention_mask is not None:
            sequence_lengths = decoder_attention_mask.sum(-1) - 1
            # else:
            #     sequence_lengths = decoder_input_ids.new_ones(decoder_input_ids.shape).sum(-1)
            last_hidden_states = hidden_states[range(sequence_lengths.shape[0]), sequence_lengths]  # bs,hn
            # judgment classification
            if self.judgment_scorer is not None:
                judgment_logits = self.judgment_scorer(last_hidden_states)
                other_outputs["judgment_logits"] = judgment_logits
            # agency classification
            if self.agency_scorer is not None:
                agency_logits = self.agency_scorer(last_hidden_states)
                other_outputs["agency_logits"] = agency_logits

            char_logits = self.char_scorer(last_hidden_states)
            other_outputs["char_logits"] = char_logits

            # # agree classification
            # agree_logits = self.agree_scorer(last_hidden_states)
            # other_outputs["agree_logits"] = agree_logits

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # 1. causal loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            batch_size, seq_len, n_classes = lm_logits.shape
            losses_lm_2d = loss_fn(lm_logits.view(-1, n_classes), labels.view(-1))  # pooling
            losses_lm_2d = losses_lm_2d.view(batch_size, seq_len)  # bs, sl
            # use macro loss
            losss_lm_1d = masked_pool(losses_lm_2d, kwargs["lm_loss_mask"], high_rank=False)  # bs
            lm_loss = losss_lm_1d.mean()
            other_outputs["lm_loss"] = lm_loss

            if "[clm]" in self.config.loss_components:
                loss = lm_loss
            else:
                loss = 0.

            if "judgment_logits" in other_outputs:
                judgment_loss = torch.nn.CrossEntropyLoss()(other_outputs["judgment_logits"], kwargs["judgment_labels"])
                other_outputs["judgment_loss"] = judgment_loss
                if "[judgment]" in self.config.loss_components:
                    loss += judgment_loss

            if "agency_logits" in other_outputs:
                agency_loss = torch.nn.CrossEntropyLoss()(other_outputs["agency_logits"], kwargs["agency_labels"])
                other_outputs["agency_loss"] = agency_loss
                if "[agency]" in self.config.loss_components:
                    loss += agency_loss

            if "char_logits" in other_outputs:
                char_loss = torch.nn.CrossEntropyLoss()(other_outputs["char_logits"], kwargs["char_labels"])
                other_outputs["char_loss"] = char_loss
                if "[char]" in self.config.loss_components:
                    loss += char_loss


            # # judgment classification
            # if "judgment_labels" in kwargs:
            #     judgment_loss = torch.nn.CrossEntropyLoss()(judgment_logits, kwargs["judgment_labels"])
            #     other_outputs["judgment_loss"] = judgment_loss
            #     if "[soft1_judgment]" in self.config.loss_components:
            #         # # choice 1: full soft
            #         soft_labels = kwargs.get("soft_judgment_labels")
            #         pred_logprob = nn.LogSoftmax(dim=-1)(judgment_logits)
            #         soft_judgment_loss = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
            #         other_outputs["soft_judgment_loss"] = soft_judgment_loss
            #         loss += soft_judgment_loss
            #     elif "[soft2_judgment]" in self.config.loss_components:
            #         # # choice 1: full soft
            #         soft_labels = kwargs.get("soft_judgment_labels")
            #         soft_weight = torch.max(soft_labels, dim=-1)[0]
            #         losses_soft_judgment = CrossEntropyLoss(reduction="none")(judgment_logits, kwargs["judgment_labels"]) * soft_weight
            #         soft_judgment_loss = torch.mean(losses_soft_judgment)
            #         other_outputs["soft_judgment_loss"] = soft_judgment_loss
            #         loss += soft_judgment_loss
            #     elif "[judgment]" in self.config.loss_components:
            #         loss += judgment_loss
            #
            # if "agree_labels" in kwargs:
            #     agree_loss = torch.nn.CrossEntropyLoss()(agree_logits, kwargs["agree_labels"])
            #     other_outputs["agree_loss"] = agree_loss
            #     if "[agree]" in self.config.loss_components:
            #         loss += agree_loss

        model_outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        for k in other_outputs:
            model_outputs[k] = other_outputs[k]

        return model_outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


MODEL_FOR_SC101_GEN_MAPPING = OrderedDict(
    [
        (BartConfig, BartForSC101Gen),
    ]
)


AutoModelForSC101Gen = auto_class_factory(
    "AutoModelForBoolScruples",
    MODEL_FOR_SC101_GEN_MAPPING,
    head_doc="bool sequence classification for scruples"
)
