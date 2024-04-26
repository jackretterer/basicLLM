import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class LyricsDataset(Dataset):
    def __init__(self, file_path, context_window=5):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.lyrics = [line.strip() for line in file if line.strip()]
        self.vocab = sorted(set(' '.join(self.lyrics).split()))
        self.vocab.append('<unk>')  # Add the '<unk>' token to the vocabulary
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.context_window = context_window
    
    def __len__(self):
        return len(self.lyrics)
    
    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        tokens = lyric.split()
        
        input_indices = []
        target_indices = []
        
        for i in range(len(tokens) - self.context_window):
            input_seq = tokens[i:i + self.context_window]
            target_word = tokens[i + self.context_window]
            
            input_seq_indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in input_seq]
            target_word_index = self.word2idx.get(target_word, self.word2idx['<unk>'])
            
            input_indices.append(input_seq_indices)
            target_indices.append(target_word_index)
        
        if len(input_indices) > 0 and len(target_indices) > 0:
            return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)
        else:
            return None
    
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
    
# Hyperparameters
hidden_dim = 128
num_layers = 2
num_heads = 4
batch_size = 16
num_epochs = 10
learning_rate = 0.001

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    input_seqs, target_seqs = zip(*batch)
    
    # Pad the input sequences
    padded_input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    
    # Pad the target sequences
    padded_target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=-100)
    
    return padded_input_seqs, padded_target_seqs

# Load the dataset
dataset = LyricsDataset('all_tswift_lyrics_cleaned.txt')
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# Instantiate the model, loss function, and optimizer
vocab_size = len(dataset.vocab)
model = SimpleTransformer(vocab_size, hidden_dim, num_layers, num_heads)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the device (CPU in this case)
device = torch.device('cpu')

# Move the model to the device
model.to(device)

# Train the model
for epoch in range(num_epochs):
    for batch in dataloader:
        if batch is None:
            continue
        
        input_seq, target_seq = batch
        optimizer.zero_grad()
        input_seq = input_seq.to(device)  # Move input to the device (CPU or GPU)
        target_seq = target_seq.to(device)  # Move target to the device (CPU or GPU)
        
        outputs = model(input_seq)
        outputs = outputs.view(-1, outputs.shape[-1])
        target_seq = target_seq.view(-1)
        
        # Create a mask to ignore padded tokens in the loss calculation
        mask = (target_seq != -100)
        
        # Apply the mask to the outputs and target sequences
        outputs = outputs[mask]
        target_seq = target_seq[mask]
        
        loss = criterion(outputs, target_seq)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the trained model
with torch.no_grad():
    test_lyrics = "So shame on "
    test_seq = torch.tensor([dataset.word2idx.get(word, dataset.word2idx['<unk>']) for word in test_lyrics.split()], dtype=torch.long)
    output = model(test_seq.unsqueeze(0))
    output_probs = torch.softmax(output[-1, -1], dim=-1)
    predicted_index = torch.multinomial(output_probs, num_samples=1).item()
    
    if predicted_index < len(dataset.vocab):
        predicted_word = dataset.vocab[predicted_index]
        predicted_lyrics = test_lyrics + ' ' + predicted_word
    else:
        predicted_lyrics = test_lyrics
        print(f"Skipping prediction. Predicted index {predicted_index} is out of range.")

    print("Input lyrics:", test_lyrics)
    print("Predicted word:", predicted_word)
    print("Predicted lyrics:", predicted_lyrics)

# Evaluate the trained model
with torch.no_grad():
    test_lyrics = "I knew you were"
    test_seq = torch.tensor([dataset.word2idx.get(word, dataset.word2idx['<unk>']) for word in test_lyrics.split()], dtype=torch.long)
    output = model(test_seq.unsqueeze(0))
    output_probs = torch.softmax(output[-1, -1], dim=-1)
    
    num_predictions = 5
    predicted_indices = torch.multinomial(output_probs, num_samples=num_predictions)
    predicted_words = [dataset.vocab[idx.item()] for idx in predicted_indices]
    predicted_probs = [output_probs[idx.item()].item() for idx in predicted_indices]
    
    print("Input lyrics:", test_lyrics)
    print("Predicted words:", predicted_words)
    print("Predicted probabilities:", predicted_probs)