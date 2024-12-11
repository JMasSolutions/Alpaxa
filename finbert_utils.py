from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

# Check for available device (MPS -> CUDA -> CPU)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news: list) -> Tuple[float, str]:
    if news:
        # Tokenize input and move tensors to the selected device
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Perform inference
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)

        # Extract sentiment and its probability
        probability = result[torch.argmax(result)].item()
        sentiment = labels[torch.argmax(result).item()]
        return probability, sentiment
    else:
        return 0.0, labels[-1]  # Default to "neutral"


if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!', 'traders were displeased!'])
    print(f"Sentiment: {sentiment}, Probability: {tensor:.4f}")
    print(f"Device in use: {device}")