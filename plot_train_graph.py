import json

import matplotlib.pyplot as plt

log_file = 'log (1).txt'
loss_values = []

with open(log_file, 'r') as f:
    for line in f:
        log_entry = json.loads(line)
        loss_values.append(log_entry['train_loss'])

plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()