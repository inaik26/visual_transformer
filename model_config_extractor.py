from transformers import AutoModel, AutoTokenizer, AutoConfig

# Replace 'model_name' with the desired model identifier
model_name = 'distilbert-base-uncased'

# Load the configuration
config = AutoConfig.from_pretrained(model_name)
print("Model Configuration:\n", config)

# Load the tokenizer (if needed)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer:\n", tokenizer)

# Load the model
model = AutoModel.from_pretrained(model_name)
print("Model Architecture:\n", model)

