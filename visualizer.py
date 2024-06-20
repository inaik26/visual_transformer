import torch
from transformers import BertModel
from torch.utils.tensorboard import SummaryWriter

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Set up TensorBoard writer
writer = SummaryWriter('runs/bert_model')

# Create dummy input
dummy_input = torch.randint(0, 1000, (1, 10))

# Trace the model
traced_model = torch.jit.trace(model, dummy_input, strict=False)

# Log the model graph
# Use the traced model instead of the original model
writer.add_graph(traced_model, dummy_input)
writer.close()

# Instructions to run TensorBoard
print("Run the following command in your terminal to start TensorBoard:")
print("tensorboard --logdir=runs")
print("Then open a web browser and go to the URL provided (usually http://localhost:6006).")
