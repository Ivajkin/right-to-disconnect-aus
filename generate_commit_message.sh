#!/bin/bash

# ===================================================
# Script: generate_commit_message.sh
# Description: Generates smart git commit messages using a local LLM.
# Requirements:
#   - Python 3.x
#   - Transformers library
#   - A compatible local LLM (e.g., FLAN-T5 Base)
# Usage:
#   ./generate_commit_message.sh
# ===================================================

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH="/Users/tim/models/flan-t5-base"      # Update with your local model path
TOKENIZER_PATH="/Users/tim/models/flan-t5-base"  # Usually same as MODEL_PATH
MAX_LENGTH=150                                  # Maximum tokens for commit message
MAX_DIFF_LINES=100                              # Maximum lines of diff to include

# -------------------------------
# Function: Check Dependencies
# -------------------------------
check_dependencies() {
    command -v git >/dev/null 2>&1 || { echo >&2 "Git is required but not installed. Aborting."; exit 1; }
    command -v python3 >/dev/null 2>&1 || { echo >&2 "Python3 is required but not installed. Aborting."; exit 1; }
    python3 -c "import transformers" >/dev/null 2>&1 || { echo >&2 "Python 'transformers' library is required but not installed. Aborting."; exit 1; }
}

# -------------------------------
# Function: Generate Commit Message
# -------------------------------
generate_commit_message() {
    # Get the full diff of staged changes
    local full_diff=$(git diff --cached)

    # Limit the diff to MAX_DIFF_LINES to prevent exceeding model input limits
    local limited_diff=$(echo "$full_diff" | head -n $MAX_DIFF_LINES)

    # Check if diff was truncated
    local diff_line_count=$(echo "$full_diff" | wc -l)
    if [ "$diff_line_count" -gt "$MAX_DIFF_LINES" ]; then
        limited_diff="${limited_diff}\n... (diff truncated)"
        echo "Warning: Diff output truncated to $MAX_DIFF_LINES lines."
    fi

    # Build the prompt with detailed diffs
    local prompt="You are an AI assistant that writes concise and descriptive git commit messages.
Here is the detailed diff of the changes made:

$limited_diff

Based on these changes, generate a descriptive and concise git commit message that explains what was changed and why."

    echo -e "Prompt:\n$prompt\n"

    # Pass the prompt to Python for generation
    commit_message=$(MODEL_PATH="$MODEL_PATH" TOKENIZER_PATH="$TOKENIZER_PATH" python3 <<EOF
import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    try:
        model_path = os.getenv('MODEL_PATH')
        tokenizer_path = os.getenv('TOKENIZER_PATH')
        if not model_path or not tokenizer_path:
            print("Model path or tokenizer path not set.", file=sys.stderr)
            sys.exit(1)

        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Prepare the prompt
        prompt = """$prompt"""

        # Tokenize and generate
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        outputs = model.generate(
            inputs,
            max_length=$MAX_LENGTH,        # Limit commit message length
            num_beams=5,                   # Use beam search for better generation
            early_stopping=True
        )

        # Decode and print the commit message
        commit_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Ensure the commit message does not include the prompt
        if commit_message.startswith(prompt):
            commit_message = commit_message[len(prompt):].strip()
        print(commit_message)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
)

    echo "$commit_message"
}

# -------------------------------
# Function: Main Workflow
# -------------------------------
main() {
    check_dependencies

    # Check for staged changes
    if git diff --cached --quiet; then
        echo "No staged changes to commit."
        exit 0
    fi

    # Generate the commit message based on the diff
    echo -e "Generating commit message..."
    commit_message=$(generate_commit_message)

    # Check if commit message was generated successfully
    if [ -z "$commit_message" ]; then
        echo "Failed to generate commit message."
        exit 1
    fi

    # Display the generated commit message
    echo "----------------------------------------"
    echo "Generated Commit Message:"
    echo "$commit_message"
    echo "----------------------------------------"

    # Prompt for confirmation
    read -r -p "Do you want to use this commit message? (y/n): " choice
    case "$choice" in
        y|Y )
            git commit -m "$commit_message"
            echo "Commit successful."
            ;;
        n|N )
            echo "Please enter your commit message manually."
            git commit
            ;;
        * )
            echo "Invalid choice. Commit aborted."
            exit 1
            ;;
    esac
}

# -------------------------------
# Execute the Main Function
# -------------------------------
main
