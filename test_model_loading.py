from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/Users/tim/models/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("/Users/tim/models/flan-t5-base")

# Example prompt
prompt = """You are an AI assistant that writes concise and descriptive git commit messages.
Here is the summary of the changes made:

Changes: 3 file(s), 141 insertion(s), 0 deletion(s)

Based on these changes, generate a descriptive and concise git commit message."""

# Tokenize and generate
inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
outputs = model.generate(
    inputs,
    max_length=100,        # Limit commit message length
    num_beams=5,           # Use beam search for better generation
    early_stopping=True
)

# Decode and print the commit message
commit_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(commit_message.strip())
