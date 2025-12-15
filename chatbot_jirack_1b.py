# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# This file is part of a project authored by CMS Manhattan. You may use, distribute, and modify
# this code under the terms of the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
# Please read <http://www.gnu.org/licenses/>.

# JiRack 1B Chatbot — fixed and improved version, December 2025

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from JiRackPyTorch_GPT5_class_1b import JiRackPyTorch
from pathlib import Path

# ============================= GENERATION SETTINGS =============================
TEMPERATURE = 0.7          # Lower = more deterministic, higher = more creative
TOP_K = 50                 # Set to 0 to disable top-k sampling
MAX_NEW_TOKENS = 120       # Maximum number of new tokens to generate per response

# ============================= PATHS =============================
# Primary path to your fine-tuned model weights
MODEL_PATH = Path("build/fine_tuning_output/epoch2/gpt_finetuned.pt")

# Fallback checkpoint paths (in order of preference)
FALLBACK_PATHS = [
    Path("checkpoints/jirack_1b_epoch3.pt"),
    Path("checkpoints/jirack_1b_epoch2.pt"),
    Path("checkpoints/jirack_1b_epoch1.pt"),
]

# Path to your custom tokenizer (must match the one used during training)
TOKENIZER_PATH = Path("/home/kgrabko/jirackkit/src/main/python/tokenizer/tokenizer.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================= Chatbot Class =============================
class Chatbot:
    def __init__(self, tokenizer_path: Path, model_weights_path: Path | None = None):
        # 1. Load custom tokenizer
        print("Loading custom tokenizer from tokenizer.json...")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = self.tokenizer.get_vocab_size()
        print(f"Tokenizer loaded successfully. Vocabulary size: {vocab_size}")

        # 2. Initialize the model
        print("Initializing JiRackPyTorch 1B model...")
        self.model = JiRackPyTorch().to(device)
        self.model.eval()

        # 3. Load trained weights
        load_path = None
        if model_weights_path and model_weights_path.exists():
            load_path = model_weights_path
        else:
            for path in FALLBACK_PATHS:
                if path.exists():
                    load_path = path
                    break

        if load_path:
            print(f"Loading model weights from: {load_path}")
            state_dict = torch.load(load_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Weights loaded successfully. Model is running on {device}.")
        else:
            print("Warning: No trained weights found. Model will use random initialization (output will be garbage).")

        print("🤖 JiRack 1B Chatbot is ready!\n")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
    ) -> str:
        # Encode the user prompt
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

        # Generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass — model returns logits only
                logits = self.model(input_ids)  # (1, seq_len, vocab_size)

                # Take logits of the last token
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k sampling
                if top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    values, indices = torch.topk(next_token_logits, top_k_val)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, indices, values)

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Optional: stop early on EOS token (if your tokenizer has one)
                eos_id = self.tokenizer.token_to_id("<|endoftext|>")
                if eos_id is not None and next_token.item() == eos_id:
                    break

        # Decode the full generated sequence
        full_output = self.tokenizer.decode(input_ids.squeeze(0).cpu().tolist())

        # Extract only the response part (remove the original prompt)
        response = full_output[len(prompt):].strip()

        # Clean up common artifacts
        response = response.replace("__eou__", "").replace("<|endoftext|>", "").strip()

        return response if response else "[No response generated]"


# ============================= Main Chat Loop =============================
def main():
    global TEMPERATURE, TOP_K, MAX_NEW_TOKENS

    # Create chatbot instance
    chatbot = Chatbot(tokenizer_path=TOKENIZER_PATH, model_weights_path=MODEL_PATH)

    print("=" * 70)
    print("🤖 JiRack 1B Chatbot Activated — December 2025")
    print(f"   Current settings → Temperature: {TEMPERATURE} | Top-K: {TOP_K} | Max tokens: {MAX_NEW_TOKENS}")
    print("   Commands:")
    print("     • Type 'exit' or 'quit' to stop")
    print("     • 'set temp=0.8' — change temperature")
    print("     • 'set k=60'     — change top-k")
    print("     • 'set max=200'  — change max new tokens")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye! Thanks for chatting.")
                break

            # Settings commands
            if user_input.lower().startswith("set temp="):
                try:
                    TEMPERATURE = float(user_input.split("=")[1])
                    print(f"🤖 Temperature updated to {TEMPERATURE}")
                    continue
                except:
                    print("🤖 Invalid value. Example: set temp=0.8")
                    continue

            if user_input.lower().startswith("set k="):
                try:
                    TOP_K = int(user_input.split("=")[1])
                    print(f"🤖 Top-K updated to {TOP_K}")
                    continue
                except:
                    print("🤖 Invalid value. Example: set k=50")
                    continue

            if user_input.lower().startswith("set max="):
                try:
                    MAX_NEW_TOKENS = int(user_input.split("=")[1])
                    print(f"🤖 Max new tokens updated to {MAX_NEW_TOKENS}")
                    continue
                except:
                    print("🤖 Invalid value. Example: set max=150")
                    continue

            # Generate and display response
            print("🤖 Thinking...")
            response = chatbot.generate_response(
                prompt=user_input + " ",  # small space helps with continuation
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )
            print(f"JiRack: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()