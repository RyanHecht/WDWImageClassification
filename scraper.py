import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb

api_key =  u'de2346d37930a57db36e02ccc155673b'
api_secret = u'c4d4b868c3878f89'

csv_path = 'VisionFinal/regions/magic_kingdom/park.csv'
coords = []
with open(csv_path, "r") as f:
    for line in f.readlines():
        coords.append([float(line.split(",")[i]) for i in range(len(line.split(",")))])
coords = np.array(coords)
max_lat = round(max(coords[:, 0]), 4)
max_long = round(max(coords[:, 1]), 4)
min_lat = round(min(coords[:, 0]), 4)
min_long = round(min(coords[:, 1]), 4)
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
bbox_coords = [max_lat, max_long, min_lat, min_long]
print(bbox_coords)
photos_search = flickr.photos.search(bbox = bbox_coords, geo_context = 2, min_upload_date = 1524009600)
with open("resutls.txt", "w") as g:
    g.write(json.dumps(photos_search))
g.close()
