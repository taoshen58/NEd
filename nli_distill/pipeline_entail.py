from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TextClassificationPipeline
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
config = AutoConfig.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

nli_pipeline = TextClassificationPipeline(model=nli_model, tokenizer=tokenizer, device=0, return_all_scores=True)


def add_special_token(t1, t2):
    return f"<s>{t1}</s></s>{t2}</s>"




