import numpy as np
import matplotlib.patches as patches
import os
import json
import common
import sys
# Labels all locations in data/labels
label_dir = "data/labels"
def fetch_regions(type):
    regions = {}
    if type == 'parks':
        with open('regions/park_labels.json', 'r') as file:
                data = json.load(file)
                for id in data:
                        regions[id] = patches.Polygon(np.genfromtxt('regions/' + data[id]['polygon'], delimiter=','))
                        
    elif type == 'lands':
        with open('regions/land_labels.json', 'r') as file:
                data = json.load(file)
                for id in data:
                        regions[id] = patches.Polygon(np.genfromtxt('regions/' + data[id]['polygon'], delimiter=','))
    
    return regions

def get_label(lat, lng, regions):
    for region in regions:
        poly = regions[region]
        if poly.contains_point((lat, lng)):
            return region
    return -1




if len(sys.argv) == 2:
        if sys.argv[1] == "stats":
                parkStats = {}
                landStats = {}
                for filename in os.listdir("data/labels"):
                        with open(label_dir + "/" + filename, 'r') as file:
                                try:
                                        data = json.load(file)
                                        park = data['labels']['park']
                                        land = data['labels']['land']
                                        parkStats[park] = parkStats.get(park, 0) + 1
                                        landStats[land] = landStats.get(land, 0) + 1
                                
                                except Exception as e:
                                        print(e)
                        print(parkStats)
                        print(landStats)
else:


        park_regions = fetch_regions('parks')
        land_regions = fetch_regions('lands')

        for label_file in os.listdir(label_dir):
                with open(label_dir + "/" + label_file, 'r') as file:
                        print(label_file)
                        data = json.load(file)
                        location = data['location']
                        point = (float(location['lat']), float(location['lng']))

                        park_label = get_label(point[0], point[1], park_regions)
                        land_label = get_label(point[0], point[1], land_regions)
                        data.update({"labels": {"park": park_label, "land": land_label}})
                        print(data)
                        common.update_label(label_dir + "/" + label_file, data)
        
    
