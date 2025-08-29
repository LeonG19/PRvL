#Code to run base llama3 with RAG
#Code to run base llama3 with RAG
import torch
from torch import cuda
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" #Change according to desired model
tokenizer = AutoTokenizer.from_pretrained(model_id)
key = "" #Provide your hugging face key here
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", token = key
)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},  # use bfloat16 for efficiency
    device_map=device, max_new_tokens=256
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Set the environment variable

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

from datasets import load_dataset

# Load the training split of the PII masking dataset
dataset = load_dataset("ai4privacy/pii-masking-300k", split="train")
train_data = dataset.filter(lambda x: x["language"] == "English")
dataset_valdiation = load_dataset("ai4privacy/pii-masking-300k", split="validation")
validation_data = dataset_valdiation.filter(lambda x: x["language"] == "English")
unmasked_text_validation = [item["source_text"] for item in validation_data]
masked_text_validation = [item["target_text"] for item in validation_data]
masked_texts   = [item["target_text"] for item in train_data]
unmasked_texts = [item["source_text"]  for item in train_data]

# build vector store once
vector_store = FAISS.from_texts(
    unmasked_texts,
    embed_model,
    metadatas=[{"masked": m} for m in masked_texts]
)


def build_masking_prompt(retrieved_docs, sequence):
    prompt = (
        "Below is a 'sentence-to-mask' and examples of 'unmasked' - 'masked' sentences . Based on the examples write a privacy protection version of 'sentence-to-mask' in the form of a 'masked-sentence'. Sensitive information should be replaced by NAME, ORGANIZATION, etc.\n"
        "Always put your response after 'masked-sentence:'\n"
        "Examples:\n"
    )
    for i, doc in enumerate(retrieved_docs, start=1):
        prompt += (
            f"Example {i}:\n"
            f"  unmasked: {doc.page_content}\n"
            f"  masked:   {doc.metadata['masked']}\n\n"
        )
    prompt += "End of examples\n"
    prompt += f"sentence-to-mask: {sequence}\n masked-sentence:"
    return prompt
def query_rag(pipe_fn, src_data, k=3):
    
        print("sample: " + str(i))
        src = src_data
        docs = vector_store.similarity_search(src, k=k)
        prompt = build_masking_prompt(docs, src)
        out = pipe(prompt)[0]["generated_text"]        
        return out
test_sentence = "ach other respectfully in their comments. yeganeh-afchar and ylhhhrmivzz90 shared additional resources related to the video's topic, enriching the discussion further. Throughout the interaction, the diverse perspectives and insights shared by individuals added depth and richness to the educational dialogue on the platform. BACKGROUND: 22:41 on December 21st, 1966"
result = query_rag(pipe, test_sentence)
print("Here is your mask sentence", result)


