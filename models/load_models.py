from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
from peft import PeftModel

def load_codet5():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    return tokenizer, model

def load_codet5_ft():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    base = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model = PeftModel.from_pretrained(base, r"\models\checkpoint-1368")
    return tokenizer, model

def load_deepseek():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        trust_remote_code=True,
        device_map="auto" 
    )
    return tokenizer, model

def load_deepseek_ft():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, r"\models\checkpoint-608")
    return tokenizer, model
