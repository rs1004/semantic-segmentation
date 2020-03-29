import pandas as pd
import json
from pathlib import Path


d = pd.read_csv(Path(__file__).parent / 'labels.csv')

labelmap = {}
for (cat_id, category), ids in d.groupby(['catId', 'category']).groups.items():
    labelmap[category] = {'id': cat_id, 'colors': d.loc[ids, ['color_r', 'color_g', 'color_b']].drop_duplicates().values.tolist()}

with open(Path(__file__).parent.parent / 'labelmap.json', 'w') as f:
    json.dump(labelmap, f, indent=2)
