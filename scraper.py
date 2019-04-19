import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb
import urllib.request
import csv

api_key =  u'de2346d37930a57db36e02ccc155673b'
api_secret = u'c4d4b868c3878f89'
flicker_url = "flickr.com/photos/"



# Given a photo list from a call to flickr.photos.search, return a dictionary containing the distributions
#    of what sizes are available for these images.
# Example result: {u'Square': 250, u'Large 1600': 239, u'Small 320': 250, u'Original': 232, u'Large': 244, u'Medium': 250, u'Medium 640': 249, u'Large Square': 250, u'Medium 800': 240, u'Small': 250, u'Large 2048': 229, u'Thumbnail': 250}
def get_size_distribution(search_results):
    all_sizes = {}
    for im in search_results:
        sizes = flickr.photos.getSizes(photo_id=im['id'])
        #print(sizes['sizes'])
        for size in sizes['sizes']['size']:
            label = size['label']
            #print(size['label'])
            if label in all_sizes:
                all_sizes[label] += 1
            else:
                all_sizes[label] = 1
    return all_sizes

# Save the Flickr image with id `image_id` of size `size` to the folder specified in `path`
# It will be saved as `path`/`image_id`.jpg
# Its lat/lng will be saved in `path`/labels/`image_id`.json
def save_image(image_id, size, path):
    sizes = flickr.photos.getSizes(photo_id=image_id)['sizes']['size']

    medium_size = next(s for s in sizes if s['label'] == size)
    urllib.request.urlretrieve(medium_size['source'], path + "/" + image_id + ".jpg")

    location = flickr.photos.geo.getLocation(photo_id=image_id)['photo']['location']
    lat_lng = {'lat': location['latitude'], 'lng': location['longitude']}
    print(lat_lng)
    update_label(path + "/labels/" + image_id + ".json", lat_lng)


def update_label(file, dict):
    with open(file, 'w') as lbl_file:
        try:
            data = json.load(lbl_file)
        except Exception as ex:
            data = {}
        
        new_data = data.update(dict)
        json.dump(data, lbl_file)


csv_path = 'regions/magic_kingdom/park.csv'
coords = []
with open(csv_path, "r") as f:
    for line in f.readlines():
        coords.append([float(line.split(",")[i]) for i in range(len(line.split(",")))])
coords = np.array(coords)

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
# with open("results.json", "w") as g:
#     g.write(json.dumps(photos_search))
# g.close()

for im in photos:
    save_image(im['id'], "Medium", "data")


