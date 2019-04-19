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
# max_lat = np.around(max(coords[:,0]), decimals=3)
# max_long = np.around(max(coords[:, 1]), decimals=3)
# min_lat = np.around(min(coords[:, 0]), decimals=3)
# min_long = np.around(min(coords[:, 1]), decimals=3)
max_lat = "{:.3f}".format(max(coords[:,0]))
max_long = "{:.3f}".format(max(coords[:, 1]))
min_lat = "{:.3f}".format(min(coords[:, 0]))
min_long = "{:.3f}".format(min(coords[:, 1]))
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
bbox_coords = [min_long, min_lat, max_long, max_lat]
print(bbox_coords)
photos_search = flickr.photos.search(bbox = ",".join(bbox_coords), min_upload_date = 1524009600)
assert(photos_search['stat'] == 'ok')
photos = photos_search['photos']['photo']
#pdb.set_trace()
with open("results.json", "w") as g:
    g.write(json.dumps(photos_search))
g.close()
for im in photos:
    print(im)
    #flickr.photos.getSizes(im['id'])