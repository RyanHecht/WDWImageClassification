import numpy as np
import matplotlib.patches as patches
import os
import json

def fetch_regions():
    for sub in [f.path for f in os.scandir("regions") if f.is_dir() ]:
        pass

def get_label(lat, lng):
    pass


mk = patches.Polygon(np.genfromtxt('regions/magic_kingdom/park.csv', delimiter=','))

label_dir = "data/labels"
for label_file in os.listdir(label_dir):
    with open(label_dir + "/" + label_file, 'r') as file:
        data = json.load(file)
        location = data['location']
        point = (float(location['lat']), float(location['lng']))

        print(point)
        print(mk.contains_point(point))
    
