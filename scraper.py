import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb

api_key =  u'de2346d37930a57db36e02ccc155673b'
api_secret = u'c4d4b868c3878f89'
flicker_url = "flickr.com/photos/"

csv_path = 'regions/magic_kingdom/park.csv'
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
photos_search = flickr.photos.search(bbox = bbox_coords)
assert(photos_search['stat'] == 'ok')
photos = photos_search['photos']['photo']
pdb.set_trace()
with open("results.json", "w") as g:
    g.write(json.dumps(photos_search))
g.close()
for im in photos:
    flickr.photos.getSizes(im['id'])