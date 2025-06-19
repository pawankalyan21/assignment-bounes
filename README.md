# assignment-bounes
# Jagatti Pawan Kalyan
# 700776779
from transformers import pipeline

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering")

# Define context and question
context = "Charles Babbage is considered the father of the computer. He invented the first mechanical computer."
question = "Who is considered the father of the computer?"

# Get prediction
result = qa_pipeline(question=question, context=context)

print(result)

```

    No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 564e9b5 (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    config.json:   0%|          | 0.00/473 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/261M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/436k [00:00<?, ?B/s]


    Device set to use cpu


    {'score': 0.9978974461555481, 'start': 0, 'end': 15, 'answer': 'Charles Babbage'}



```python
from transformers import pipeline

# Use a specific QA model
qa_pipeline_custom = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = "Charles Babbage is considered the father of the computer. He invented the first mechanical computer."
question = "Who is considered the father of the computer?"

# Get prediction
result_custom = qa_pipeline_custom(question=question, context=context)

print(result_custom)


    config.json:   0%|          | 0.00/571 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/496M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/79.0 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/772 [00:00<?, ?B/s]


    Device set to use cpu


    {'score': 0.9878972172737122, 'start': 0, 'end': 15, 'answer': 'Charles Babbage'}



```python
# Custom context
context = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is known for the iPhone and Mac computers."

# Question 1
q1 = "Who founded Apple Inc.?"
result1 = qa_pipeline_custom(question=q1, context=context)

# Question 2
q2 = "What products is Apple known for?"
result2 = qa_pipeline_custom(question=q2, context=context)

print("Answer 1:", result1)
print("Answer 2:", result2)

```

    Answer 1: {'score': 0.9786236882209778, 'start': 26, 'end': 69, 'answer': 'Steve Jobs, Steve Wozniak, and Ronald Wayne'}
    Answer 2: {'score': 0.7343839406967163, 'start': 108, 'end': 132, 'answer': 'iPhone and Mac computers'}



```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
img_dim = 28 * 28
label_dim = 10
noise_dim = 100
batch_size = 128
epochs = 50
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img, self.label_emb(labels)], dim=1)
        return self.model(x)

# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Optimizers and loss
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for real_imgs, labels in loader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)
        labels = labels.to(device)

        # Real and fake targets
        real_targets = torch.ones(batch_size, 1).to(device)
        fake_targets = torch.zeros(batch_size, 1).to(device)

        # ========== Train Discriminator ==========
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
        fake_imgs = G(z, fake_labels)

        d_real = D(real_imgs, labels)
        d_fake = D(fake_imgs.detach(), fake_labels)

        d_loss = criterion(d_real, real_targets) + criterion(d_fake, fake_targets)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ========== Train Generator ==========
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
        gen_imgs = G(z, gen_labels)
        d_gen = D(gen_imgs, gen_labels)

        g_loss = criterion(d_gen, real_targets)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

# ========== Generate digits for labels 0-9 ==========
def show_generated_digits():
    G.eval()
    z = torch.randn(10, noise_dim).to(device)
    labels = torch.arange(0, 10).to(device)
    with torch.no_grad():
        gen_imgs = G(z, labels).view(-1, 1, 28, 28)
    gen_imgs = gen_imgs.cpu() * 0.5 + 0.5  # Denormalize

    grid = torch.cat([img.squeeze(0) for img in gen_imgs], dim=1)
    plt.imshow(grid, cmap="gray")
    plt.title("Generated digits from labels 0 to 9")
    plt.axis('off')
    plt.show()
