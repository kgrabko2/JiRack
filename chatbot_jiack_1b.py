# Copyright (c) 2025 CMS Manhattan
# All rights reserved.
# Author: Konstantin Vladimirovich Grabko
# Email: grabko@cmsmanhattan.com
# Phone: +1(516)777-0945
#
# This file is part of a project authored by CMS Manhattan. You may use, distribute, and modify
# this code under the terms of the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007.
# Please read <http://www.gnu.org/licenses/>.

# Fine tune JiRackPyTorch 1B — final clean version, December 2025

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from JiRackPyTorch_GPT5_class_1b import JiRackPyTorch # Используем тот же импорт, что и в fine_tune.py
import os
from pathlib import Path 

# ============================= НАСТРОЙКИ ГЕНЕРАЦИИ =============================
# Temperature: Чем ниже, тем более консервативны и предсказуемы ответы.
# Начните с 0.7. Если модель повторяется, повысьте до 0.8.
TEMPERATURE = 0.7 

# Top-K: Ограничивает выборку K наиболее вероятными токенами.
# Начните с 50. Увеличивайте, если ответы слишком скучные.
TOP_K = 50          

# Max Length: Максимальное количество генерируемых токенов за раз
MAX_LENGTH = 120    

# ============================= ПУТИ =============================
#LAST_TRAINED_PATH = Path("models/gpt_last_trained.pt")
LAST_TRAINED_PATH = Path("build/fine_tuning_output/epoch2/gpt_finetuned.pt")
#FINAL_OUTPUT_DIR = Path("build/fine_tuning_output/final")
FINAL_OUTPUT_DIR = Path("build/fine_tuning_output/epoch2/gpt_finetuned.pt")
MODEL_SAVE_NAME = "gpt_finetuned.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================= КЛАСС Chatbot =============================
class Chatbot:
    def __init__(self, model_path):
        # 1. Токенизатор
        print("Loading standard tokenizer (gpt2)...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. Модель
        print("Initializing model...")
        self.model = JiRackPyTorch().to(device)
        self.model.eval()

        # Поиск последних весов: сначала финальная папка, потом last_trained
        load_path = None
        if (FINAL_OUTPUT_DIR / MODEL_SAVE_NAME).exists():
            load_path = FINAL_OUTPUT_DIR / MODEL_SAVE_NAME
            print(f"Weights for Epoch 50 found. Loading and moving to {device}...")
        elif model_path.exists():
            load_path = model_path
            print(f"Loading weights from {load_path} and moving to {device}...")
        
        if load_path:
            self.model.load_state_dict(torch.load(load_path, map_location=device))
        else:
            print("Warning: No trained weights found. Using initialized model.")
            
        print(f"Model successfully loaded on {device} and ready for chat!")

    def generate_response(self, prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, top_k=TOP_K):
        # Токенизируем ввод
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Запускаем генерацию
        with torch.no_grad():
            for _ in range(max_length):
                # Пропускаем через модель
                logits, _ = self.model(input_ids)
                
                # Берем только логиты для последнего токена
                next_token_logits = logits[:, -1, :] 
                
                # Применяем температуру
                next_token_logits = next_token_logits / temperature
                
                # Применяем Top-K сэмплирование
                if top_k > 0:
                    # Отсекаем все токены, кроме TOP_K самых вероятных
                    values, indices = torch.topk(next_token_logits, top_k)
                    # Создаем маску для исключения остальных токенов
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, indices, values)

                # Преобразуем логиты в вероятности и сэмплируем следующий токен
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                # Добавляем сгенерированный токен к входным данным
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Проверяем, если сгенерирован токен конца диалога (__eou__) или конца текста (EOS)
                generated_token = self.tokenizer.decode(next_token.squeeze().item())
                if "__eou__" in generated_token or next_token.squeeze().item() == self.tokenizer.eos_token_id:
                    break

        # Декодируем всю последовательность, обрезая исходный запрос
        output = self.tokenizer.decode(input_ids.squeeze().tolist())
        
        # Убираем исходный промт
        response = output[len(prompt):].strip()
        
        # Убираем токен конца диалога, если он остался в конце
        response = response.replace("__eou__", "").strip()

        return response

def main():
    # === КОРРЕКТИРОВКА ОШИБКИ: Объявляем глобальные переменные в начале функции ===
    global TEMPERATURE, TOP_K
    
    chatbot = Chatbot(LAST_TRAINED_PATH)

    print("\n" + "="*60)
    print(f"🤖 CHATBOT ACTIVATED (PPL 2.6 / Temperature {TEMPERATURE} / Top-K {TOP_K})")
    print("Enter 'exit' or 'quit' to quit. Use 'set temp=0.x' or 'set k=N' to change settings.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input(">>> You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            # Команды управления параметрами (опционально)
            if user_input.lower().startswith('set temp='):
                try:
                    # Теперь мы можем присваивать значение напрямую, так как они объявлены глобальными выше
                    TEMPERATURE = float(user_input.split('=')[1].strip())
                    print(f"🤖 Temperature set to {TEMPERATURE}")
                    continue
                except ValueError:
                    print("🤖 Invalid temperature value. Use 'set temp=0.x'.")
                    continue
            
            if user_input.lower().startswith('set k='):
                try:
                    # Теперь мы можем присваивать значение напрямую, так как они объявлены глобальными выше
                    TOP_K = int(user_input.split('=')[1].strip())
                    print(f"🤖 Top-K set to {TOP_K}")
                    continue
                except ValueError:
                    print("🤖 Invalid K value. Use 'set k=N' (e.g., set k=50).")
                    continue

            print("...Generating...")
            response = chatbot.generate_response(user_input)
            print(f"🤖 Model: {response}\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    from pathlib import Path
    main()