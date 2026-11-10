import torch
import torch.nn.functional as F
import random


words = open('data/names.txt').read().splitlines()

chars = sorted(set(''.join(words)))

# This is a "dict comprehension": it builds a dictionary in one line.
# enumerate(chars) gives pairs (index, item): (0, 'a'), (1, 'b'), ...
# JS equivalent:
#   Object.fromEntries(chars.map((ch, i) => [ch, i + 1]))
# We use i + 1 so the letters get 1..26 and number 0 stays free for "."
stringToInteger = {ch: i+1 for i, ch in enumerate(chars)}

#marks the start and end of every name.
stringToInteger['.'] = 0

IntegerToString = {i: ch for ch, i in stringToInteger.items()}

# total number of tokens the model can see: 26 letters + 1 for "." = 27
vocab_size = len(stringToInteger)

#this is how many letters will model see while creating name
block_size = 3

def build_dataset(words):
    contexts = [] #inputs, each item is a list of 3 letter numbers
    targets = [] # outputs: the number of the leter than comes next
    
    for word in words:
        # [0] -> ['.'], so every name starts with empty window.
        context = [0] * block_size
        
        # loop over each letter of the name.
        for char in word + '.':
            next_index = stringToInteger[char]
            contexts.append(context) #three letters that model sees
            targets.append(next_index) #the letter it should predict
            
            #slide the window, drop oldest letter, add new one we just saw.
            context = context[1:] +[next_index]
    # a list of lists which becomes a 2D tensor, shape: (number of letters in all names, 3)
    return torch.tensor(contexts), torch.tensor(targets)

#shuffle names first, otherwise split would be the file's order
#random.seed fixes the shuffle, so it's same on every run.
#shuffle changes the list in place, so we don't need to assign it to a new variable.
random.seed(42)
random.shuffle(words)

#int() cuts off the decimals.
train_end = int(0.8 * len(words)) #index where train stops
val_end = int(0.9 * len(words)) #index where val stops, test starts

train_contexts, train_targets = build_dataset(words[:train_end])
val_contexts, val_targets = build_dataset(words[train_end:val_end])
test_contexts, test_targets = build_dataset(words[val_end:])

print(f"train: {train_contexts.shape} {train_targets.shape}")
print(f"val: {val_contexts.shape} {val_targets.shape}")

generator = torch.Generator().manual_seed(42)

# how many numbers describe each letter
embedding_size = 10

# 27 rows (one per token) * 10 columns (the numbers describing that token)
embedding_table = torch.randn((vocab_size, embedding_size), generator=generator)


hidden_size = 200
input_size = block_size * embedding_size # 3 * 10 = 30 numbers go into the hidden layer

# 30 inputs = 200 neurons, 0.2 makes the start value small, so tanh doesn't get pushed to its flat parts too quickly.
hidden_weights = torch.randn((input_size, hidden_size), generator=generator) * 0.2
hidden_bias = torch.zeros(hidden_size)

# output layer: 200 neurons -> 27 logits
# 0.01 makes the first logits almost 0 -> the first predictions are almost uniform, so the model has to learn from scratch.
output_weights = torch.randn((hidden_size, vocab_size), generator=generator) * 0.01
output_bias = torch.zeros(vocab_size)

# collect all parameters in one list so we can loop through them during training
parameters = [embedding_table, hidden_weights, hidden_bias, output_weights, output_bias]
for parameter in parameters:
    parameter.requires_grad = True
    
#numel() = number of elemets, how many nubmers are in the tensor.
print(sum(parameter.numel() for parameter in parameters)) # 11,897 learnable numbers

# ---- one-off forward pass (snippet 4), kept for reference. The forward() function replaces it ----
# # 32 random row numbers between 0 and the number of training examples
# batch_indices = torch.randint(0, train_contexts.shape[0], (32,), generator=generator)
# batch_contexts = train_contexts[batch_indices] # (32, 3)
# batch_targets = train_targets[batch_indices] # (32,)
# 
# embeddings = embedding_table[batch_contexts] # (32, 3, 10) = 32 examples, 3 letters each, 10 numbers per letter
# 
# #.view() reshapes without changin the numbers. glue the 3 letter vectors side by side.
# # (32, 3, 10) -> (32, 30) the -1 means 'work this dimensions out yourself.
# flat_embeddings = embeddings.view(-1, input_size)
# 
# # hidden layer: multiply, add bias, squeeze tanh -> every value between -1 and 1
# hidden = torch.tanh(flat_embeddings @ hidden_weights + hidden_bias)
# 
# logits = hidden @ output_weights + output_bias
# loss = F.cross_entropy(logits, batch_targets)
# 
# print(embeddings.shape, flat_embeddings.shape, hidden.shape, logits.shape)
# print(f"starting loss: {loss.item():.4f}")

def forward(contexts):
    embeddings = embedding_table[contexts]
    flat_embeddings = embeddings.view(-1, input_size)
    hidden = torch.tanh(flat_embeddings @ hidden_weights + hidden_bias)
    logits = hidden @ output_weights + output_bias
    return logits


batch_size = 32
training_steps = 200_000

for step in range(training_steps):
    # 32 random row numbers -> 32 random training examples
    batch_indices = torch.randint(0, train_contexts.shape[0], (batch_size,), generator=generator)
    batch_contexts = train_contexts[batch_indices]
    batch_targets = train_targets[batch_indices]
    
    logits = forward(batch_contexts)
    loss = F.cross_entropy(logits, batch_targets)
    
    for parameter in parameters:
        parameter.grad = None # clears old gradients for every parameter
    loss.backward() #fills .grad on all 5 parameters, including the embedding table
    
    #big steps for first half, then 10* smaller for fine-turn.
    learning_rate = 0.1 if step < 100_000 else 0.01
    for parameter in parameters:
        parameter.data -= learning_rate * parameter.grad
        
    if step % 10_000 == 0:
        print(f"step {step:6d} / {training_steps}: loss {loss.item():.4f}")
        
        
@torch.no_grad()
def evaluate(contexts, targets):
    logits = forward(contexts)
    return F.cross_entropy(logits, targets).item()
print(f"train loss: {evaluate(train_contexts, train_targets):.4f}")
print(f"val loss: {evaluate(val_contexts, val_targets):.4f}")

@torch.no_grad()
def generate(temperature=1.0):
    context = [0] * block_size
    generated_chars = []
    while True:
        #forward() expects a batch fo shape (n,3) so wrap one context in a list -> shape (1,3)
        logits = forward(torch.tensor([context])) # (1,27)
        probs = F.softmax(logits / temperature, dim=1)
        next_index = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if next_index == stringToInteger['.']:
            break
        generated_chars.append(IntegerToString[next_index])
        context = context[1:] + [next_index]
        
    return "".join(generated_chars)

for _ in range(20):
    print(generate())

known_names = set(words)
generated_names = [generate() for _ in range(1000)]

copied_count = sum(name in known_names for name in generated_names)
print(f"{copied_count} of 1000 generated names already exist in the dataset")
print(f"that's {copied_count / 10:.1f} %")