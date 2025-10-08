import torch
import torch.nn as nn
import torch.optim as optim

# tiny bag-of-words model (toy)
class TinyText(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=16, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 16), nn.ReLU(), nn.Linear(16, num_classes))

    def forward(self, x):
        # x is [batch, tokens], we take mean embedding
        x = self.embed(x).mean(dim=1)
        return self.fc(x)

def main():
    model = TinyText()
    # toy training: random data — only to produce a model file
    dummy_x = torch.randint(0, 999, (64, 8))
    dummy_y = torch.randint(0, 1, (64,))
    optim_ = optim.Adam(model.parameters())
    lossfn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        optim_.zero_grad()
        out = model(dummy_x)
        loss = lossfn(out, dummy_y)
        loss.backward()
        optim_.step()
    torch.save(model.state_dict(), "models/tiny_text.pth")
    print("Saved models/tiny_text.pth")

if __name__ == "__main__":
    main()
