import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from SmartKVCache import SmartKVDynamicCache, GateLoader

model_id = "Qwen/Qwen3-0.6B"

# 2. Load the tokenizer associated with the model
# AutoTokenizer automatically selects the correct tokenizer class
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)

# 3. Load the model with pretrained weights
# AutoModelForCausalLM automatically selects the correct model architecture
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

# Optional: set model to evaluation mode
model.eval()

cache = SmartKVDynamicCache(
            window_size=50,
            sink_size=4,
            cache_budget=50,
            num_layers=len(model.model.layers),
            num_kv_heads=model.config.num_key_value_heads,
            device=torch.device('mps'),
            gate_loader=GateLoader("models/gates_regression_v1")
        )

# cache.set_cache_should_prune(True)

prompt = (
    f"You're playing a word puzzle in a competition. I give you two clues: Power-Tool, Sea-Creature. "
    "You must guess two rhyming words, one related to each clue. "
    "What are the rhyming words?"
)

# 4. Prepare input text
messages = [
    {"role": "user", "content": prompt},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Convert text to PyTorch tensors

# 5. Generate text
# Use torch.no_grad() for inference to save memory and computation
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        num_return_sequences=1,
        past_key_values=cache
    )

# 6. Decode and print the generated output
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(generated_text)