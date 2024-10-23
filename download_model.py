from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model name (FLAN-T5 Base)
model_name = "google/flan-t5-base"

# Download the model and tokenizer from Hugging Face
print(f"Downloading model and tokenizer: {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define local save path
save_path = "/Users/tim/models/flan-t5-base"

# Save them to your local path
print(f"Saving model and tokenizer to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer successfully saved to {save_path}")
