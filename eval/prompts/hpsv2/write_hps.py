import os
import json

for name in ['anime', 'concept-art', 'paintings', 'photo']:
    with open(name + '.json') as f:
        data = json.load(f)
    with open(name, 'w') as f:
        for line in data:
            f.write(line + '\n')
