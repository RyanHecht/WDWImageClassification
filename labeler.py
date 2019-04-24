import numpy as np
import matplotlib.patches as patches
import os
import json
import common

# Labels all locations in data/labels






park_regions = common.fetch_regions('parks')
land_regions = common.fetch_regions('lands')

label_dir = "data/labels"
for label_file in os.listdir(label_dir):
    with open(label_dir + "/" + label_file, 'r') as file:
        print(label_file)
        data = json.load(file)
        location = data['location']
        point = (float(location['lat']), float(location['lng']))

        park_label = common.get_label(point[0], point[1], park_regions)
        land_label = common.get_label(point[0], point[1], land_regions)
        data.update({"labels": {"park": park_label, "land": land_label}})
        print(data)
        common.update_label(label_dir + "/" + label_file, data)
        
    
