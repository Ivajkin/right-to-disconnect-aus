from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/Users/tim/models/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("/Users/tim/models/flan-t5-base")

# Example prompt with detailed diffs
prompt = """You are an AI assistant that writes concise and descriptive git commit messages.
Here is the detailed diff of the changes made:

diff --git a/sample.py b/sample.py
index e69de29..0d1d7fc 100644
--- a/sample.py
+++ b/sample.py
@@ -0,0 +1,5 @@
+def add(a, b):
+    return a + b
+
+result = add(2, 3)
+print(result)

Based on these changes, generate a descriptive and concise git commit message that explains what was changed and why."""

# Tokenize and generate
inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
outputs = model.generate(
    inputs,
    max_length=150,        # Limit commit message length
    num_beams=5,           # Use beam search for better generation
    early_stopping=True
)

# Decode and print the commit message
commit_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Ensure the commit message does not include the prompt
if commit_message.startswith(prompt):
    commit_message = commit_message[len(prompt):].strip()
print(f"Generated Commit Message: {commit_message}")
