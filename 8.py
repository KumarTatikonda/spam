import numpy as np

samples = [
	"Jupiter has 79 known moons .",
	"Neptune has 14 confirmed moons !",
]

token_index = {}
counter = 0
for sample in samples:
	for considered_word in sample.split():
		if considered_word not in token_index:
			counter += 1
			token_index[considered_word] = counter

print("Token Index:")
print(token_index)

max_length = 6
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
	for j, considered_word in enumerate(sample.split()):
		if j < max_length:
			index = token_index[considered_word]
			results[i, j, index] = 1

print("\nOne-Hot Encoded Tensor:")
print(results)