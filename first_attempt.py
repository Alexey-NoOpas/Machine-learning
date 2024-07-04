import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # кол-во независимых последовательностей, обрабатываемых одновременно
block_size = 8 # максимальная длина контекста для предсказания токена
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2 # показатель 3e-4, но для нашей нейросети это подходит т.к. она мала
device = 'cuda' if torch.cuda.is_available() else 'cpu' # возможность запускать на видеоадаптере
eval_iters = 200
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# все уникальные символы, представленные в тексте
chars = sorted(list(set(text)))
vocab_size = len(chars)
# создаем сопоставление символов с целыми числами, создаем простой токенизатор
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # шифровщик: принимает строку, выводит список целых чисел(токенов)
decode = lambda l: ''.join([itos[i] for i in l]) # дешифровщик: принимает список целых чисел, выводит строку

# Обучение и валидация фрагментов
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # первые 90% - обучение, остальное - проверка(валидация)
train_data = data[:n]
val_data = data[n:]

# загрузка данных
def get_batch(split):
    # генерация небольшого набора входных данных "x" и целей "y"
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# простейшая биграм модель
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # каждый токен напрямую считывает логиты для следующего токена из таблицы поиска
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx(индекс) и targets(цели) оба (B,T) тензоры чисел(векторы)
        logits = self.token_embedding_table(idx) # (B,T,C) B - batch(набор), T - time(время), C - channel(канал)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # из 3х мерного массива делаем 2 мерный массив (B, T, C)--->(B*T, C), иначе .cross_entropy() вызовет ошибку
            targets = targets.view(B*T) # делаем одномерный массив из 2 мерного (B, T)--->(B*T), причина этого та же(параметры )
            loss = F.cross_entropy(logits, targets) # с помощью перекрестной энтропии вычисляем, насколько хорошо мы предсказываем следующий токен 

        return logits, loss

    def generate(self, idx, max_new_tokens): # idx - текущий контекст некоторых символов в некотором наборе, задача генератора - увеличить до (B,T+1), (B,T+2) и тд в измерении B(batch) и T(time)
        # idx это (B, T) массив индексов в текущем контексте
        for _ in range(max_new_tokens):
            # получаем предсказание токенов
            logits, loss = self(idx)
            # сосредоточиться только на последнем временном шаге
            logits = logits[:, -1, :] # становится (B, C)
            # применяем функцию softmax(преобразование вектора значений в вероятностное распределение, сумма = 1) 
            probs = F.softmax(logits, dim=-1) # (B, C)
            # выборка из распределения
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # добавить индекс выборки в текущую последовательность
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# создаем оптимизатор PyTorch
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# простейшая процедура обучения
for iter in range(max_iters):

    # время от времени оцениваем потери на наборах обучения и валидации.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # выборка массива данных
    xb, yb = get_batch('train')

    # оценить потери
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#  генерировать из модели
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))