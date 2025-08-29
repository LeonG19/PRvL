#Code to run llama3 fine tuned from model checkpoint
#!/usr/bin/env python
# LLama3_Test.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
# model id from huggingface, make sure it matches the model you will use lora weights 
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# where you saved adapter + tokenizer make sure you put the right direcotry where you saved the lora weights
SAVE_DIR   = "./model_checkpoints/llama3-8b-pii-lora"       

# 1) Load the tokenizer (with your added PII placeholder tokens)
key = "" #You need to provide your huggingface key in this space
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=key)

#Example dataset to test model
dataset_valdiation = load_dataset("ai4privacy/pii-masking-300k", split="validation")
validation_data = dataset_valdiation.filter(lambda x: x["language"] == "English")
unmasked_text_validation = [item["source_text"] for item in validation_data]
masked_text_validation = [item["target_text"] for item in validation_data]

# 2) Load the base LLaMA-3 instruct model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    token=key
)

# ■■■ Inference helper ■■■
def mask_pii(text: str) -> str:
    prompt = ("Below is a sentence-to-mask. Sensitive information in the sentence should be replaced by placeholders like [NAME], [EMAIL], [DATE], etc. (this are examples).\n Write:\n (1) a privacy-protected version of the sentence-to-mask.\n\n"
        "Only generate the privacy-protected version after 'masked-sentence',then stop.\n"
        "sentence-to-mask\n"
        + f"{text}\n\n" +
        "(1) a privacy-protected version of the sentence, masked-sentence: ")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    # strip off the prompt tokens
    gen_ids = outputs[0][ inputs["input_ids"].shape[1] : ]
    # decode (keeping [NAME], etc., but removing pad/eos)
    return tokenizer.decode(gen_ids, skip_special_tokens=False)

#Usage example, here you can replace with your own sentence
test_sentence = "ach other respectfully in their comments. yeganeh-afchar and ylhhhrmivzz90 shared additional resources related to the video's topic, enriching the discussion further. Throughout the interaction, the diverse perspectives and insights shared by individuals added depth and richness to the educational dialogue on the platform. BACKGROUND: 22:41 on December 21st, 1966"
mask_sentence = mask_pii(test_sentence)
print("Here is your mask sentence", mask_sentence)


