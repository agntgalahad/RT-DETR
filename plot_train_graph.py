import json

import matplotlib.pyplot as plt

log_file = 'modified_log.txt'
loss_values = []
epoch = []
map_values = []

with open(log_file, 'r') as f:
    for line in f:
        
        log_entry = json.loads(line)
        map_values.append(log_entry['test_coco_eval_bbox'][1] * 100)
        loss_values.append(log_entry['train_loss'])
        epoch.append(log_entry['epoch'])

plt.plot(epoch, loss_values)
plt.plot(epoch, map_values)
plt.legend(['Loss', 'AP @ IoU=0.5'])
plt.xlabel('Epoch')
plt.title('Training Loss and Average Precision (AP) vs Epoch')
plt.show()