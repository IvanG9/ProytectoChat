import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Cargar el modelo y tokenizer desde carpeta local
model_path = "models/gpt2_news"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    bos_token='<|startoftext|>',
    eos_token='<|endoftext|>',
    pad_token='<|pad|>'
)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_news_text(prompt, max_length=200, temperature=1.0):
    prompt_formatted = f"<|startoftext|><|news|> {prompt}"
    input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(device)

    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_length=max_length,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

