import torch
import math 
import torch.nn.functional as F


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

#stoi = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '.': 0}
#itos = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}

# a 27 by 27 tensor of zeros, dtype for whole numbers
tensor = torch.zeros((27, 27), dtype=torch.int32)

for word in words:
    # result: ['.', 'e', 'm', 'm', 'a', '.']
    characters = ['.'] + list(word) + ['.']
    # chs[1:] is the same list without its first item: ['e', 'm', 'm', 'a', '.']
    # zip() walks two lists side by side and gives pairs: ('.', 'e'), ('e', 'm'), ('m', 'm'), ('m', 'a'), ('a', '.') - called bigrams
    for character1, character2 in zip(characters, characters[1:]):
        # Turn both letters into numbers, then add 1 to that cell of the table.
        # N[row, col] is how you index a 2D tensor (JS would be N[row][col])
        tensor[stringToInteger[character1], stringToInteger[character2]] += 1

# smoothing: a pair that never appeared now has count of 1 instead of 0, because log(0) is undefined. 
probabilities = (tensor + 1).float()
# - sum(dim=1) adds horizontally, along each row: 1+2+3, 4+5+6, 7+8+9 → one total per row
probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

numberGenerator = torch.Generator().manual_seed(42)
for _ in range(10):
    currentLetterAsANumber = 0 # 0 = "." = start of a name
    lettersOut = [] # # the letters of the name we're building
    while True:
        #picks a random index, weighted by the probabilities in taht row. if a has 0.13 chance, then it will be picked 13% of the time. its sampling
        #num_samples=1 -> pick one, item() -> plain python int instead of a tensor.
        currentLetterAsANumber = torch.multinomial(probabilities[currentLetterAsANumber], num_samples=1, generator=numberGenerator).item()
        if currentLetterAsANumber == 0:
            break # if we hit the end-of-name token, stop building this name
        lettersOut.append(IntegerToString[currentLetterAsANumber])
    print(''.join(lettersOut)) # ['e', 'm', 'm', 'a'] -> "emma"

total_log_prob = 0.0
pair_count = 0

for word in words:
    characters = ['.'] + list(word) + ['.']
    for current_char, next_char in zip(characters, characters[1:]):
        current_index = stringToInteger[current_char]
        next_index = stringToInteger[next_char]
        #see the probability did the model give to the next letter that really came next?
        prob = probabilities[current_index, next_index]
        #it's a natural logarithm. it gives a negative number for 0 < prob < 1
        total_log_prob += torch.log(prob)
        pair_count += 1

# Average, then flip the sign → a positive "loss", lower is better
negative_log_likelihood = -total_log_prob / pair_count
#f"..." is Python's template string, like JS's `NLL: ${...}`. :.4f means "show 4 decimals".
print(f"NLL: {negative_log_likelihood.item():.4f}")

# If every token has probability 1/27, every pair gets the same log-probability,
# so the average is just that one value.
baseline_nll = -math.log(1 / vocab_size)
print(f"Baseline NLL (pure guessing): {baseline_nll:.4f}")   # about 3.30





#neural networkds

input_list = []
target_list = []

for word in words:
    chars = ['.'] + list(word) + ['.']
    for current_char, next_char in zip(chars, chars[1:]):
        input_list.append(stringToInteger[current_char])
        target_list.append(stringToInteger[next_char])
        
#turn plain python list into tensors, for fast math
inputs = torch.tensor(input_list)
targets = torch.tensor(target_list)
pair_count = inputs.shape[0] # size of each dimension of the tensor. shape[0] is the number of rows, shape[1] is the number of columns.

#one-hot encoding: turn a single number into a vector of 0s and 1s, with a 1 in the position of that number.

inputs_one_hot = F.one_hot(inputs, num_classes=vocab_size).float()


#the whole model is 27 by 27 weight matrix. there's no hidden layer and no bias.
generator = torch.Generator().manual_seed(42)

#vocab_size is 27, normal distribution, requires_grad=True means to track the gradient of this tensor during backpropagation, so we can update it later.
weights = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)

# @ is a matrix multiplication.
# (228146, 27) @ (27, 27) → (228146, 27): 27 raw scores for every pair = the LOGITS
logits = inputs_one_hot @ weights
print(logits.shape)

#exp() makes every score positive, because logits can be negative.
fake_counts = logits.exp()

#divide each row by its sum, so every row adds up to 1. this is the SOFTMAX function.
probs = fake_counts / fake_counts.sum(dim=1, keepdim=True)

#for every pair, take the prob the model gave to the real next letter.
real_next_probs = probs[torch.arange(pair_count), targets]

#negative log likelihood: take the log of those probabilities, flip the sign, then average over all pairs.
loss = -real_next_probs.log().mean()
print(f"loss (by hand): {loss.item():.4f}")

#The same thing in one call:
# F.cross_entropy does softmax + picking the real letter + log + mean + minus, all at once.
# It takes LOGITS, not probabilities. Never softmax before it (Common Mistake #5).
# loss = F.cross_entropy(logits, targets)
# print(f"loss (cross_entropy): {loss.item():.4f}")


learning_rate = 50

for step in range(200):
    # forward pass: weights change every step, so logits and loss have to be recomputed every step. - aka computing loss with current weights
    logits = inputs_one_hot @ weights
    loss = F.cross_entropy(logits, targets)
    
    #backward pass: compute the gradients. pytorch adds new gradients to old ones instead of replacing them, so we clear the old ones first. None means no gradient. it's slightly faster than filling with zeros
    weights.grad = None
    
    #autograd: pytorch walks back through every calculation that used weights, and computes the gradient of loss with respect to weights. it stores that gradient in weights.grad
    loss.backward()
    
    #updatestep every weight against its gradient.
    # learning_rate is a hyperparameter that controls how big a step we take in the direction of the gradient. too small = slow learning, too big = overshoot and diverge.
    weights.data -= learning_rate * weights.grad
    
    #print every 10 steps, so we can see if the loss is going down. .item() turns a 1-element tensor into a plain python number.
    if step % 10 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
        
print("Final loss:", loss.item())

generator = torch.Generator().manual_seed(42)

# pytorch stops recording calculations. we're only looking at the weights.
with torch.no_grad():
    for _ in range(10):
        current_index = stringToInteger['.']
        generated_name = []
        while True:
            probs = F.softmax(weights[current_index], dim=0)
            current_index = torch.multinomial(probs, num_samples=1, generator=generator).item()
            if current_index == stringToInteger['.']:
                break
            generated_name.append(IntegerToString[current_index])
        print(''.join(generated_name))
    
    
def main():
    print("Hello from name-generator!")


if __name__ == "__main__":
    main()
