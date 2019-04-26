import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb
import urllib.request
from urllib.error import HTTPError
import csv
import common
import base64
import hashlib
import hmac
import sys

with open("config.json", 'r') as file:
        config = json.load(file)

api_key =  config['flickr']['api_key']
api_secret = config['flickr']['api_secret']
google_key = config['google_maps']['peter']['api_key']
flicker_url = "flickr.com/photos/"



def sign_url(input):
    #secret = "qpiUWhWBpyuWTwkA5_gldQ3FNg8="
    secret = config['google_maps']['peter']['secret']
    url = urllib.parse.urlparse(input)
    url_to_sign = url.path + "?" + url.query

    decoded_key = base64.urlsafe_b64decode(secret)

    signature = hmac.new(decoded_key, url_to_sign.encode('utf-8'), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    return original_url + "&signature=" + encoded_signature.decode('utf-8')

    
# Given a photo list from a call to flickr.photos.search, return a dictionary containing the distributions
#    of what sizes are available for these images.
# Example result: {u'Square': 250, u'Large 1600': 239, u'Small 320': 250, u'Original': 232, u'Large': 244, u'Medium': 250, u'Medium 640': 249, u'Large Square': 250, u'Medium 800': 240, u'Small': 250, u'Large 2048': 229, u'Thumbnail': 250}
def get_size_distribution(search_results):
    all_sizes = {}
    for im in search_results:
        sizes = flickr.photos.getSizes(photo_name=im['name'])
        #print(sizes['sizes'])
        for size in sizes['sizes']['size']:
            label = size['label']
            #print(size['label'])
            if label in all_sizes:
                all_sizes[label] += 1
            else:
                all_sizes[label] = 1
    return all_sizes

# https://developers.google.com/maps/documentation/streetview/intro
def get_streetview_in_bounding_box(min_lat, min_long, max_lat, max_long):
    streetview_query = "https://maps.googleapis.com/maps/api/streetview"
    metadata_query = streetview_query + "/metadata"
    images = 0
    for lat in np.linspace(min_lat, max_lat, 100):
        for lng in np.linspace(min_long, max_long, 100):
            query_options = "?key=" + google_key + "&location=" + str(lat) + "," + str(lng)
            try:
                with urllib.request.urlopen(sign_url(metadata_query + query_options)) as metadata_url:
                    data = json.loads(metadata_url.read().decode())
                    print(data)
                    if data['status'] == "OK":
                        images += 1
                        streetview_query_options = query_options + "&size=600x400"
                        headings = [0, 90, 180, 270]
                        pitches = [0, 30, -30]
                        real_lat = data['location']['lat']
                        real_lng = data['location']['lng']
                        print(str(real_lat) + ", " + str(real_lng))
                        for heading in headings:
                            for pitch in pitches:
                                try:
                                    image_name = data['pano_id'] + "_" + str(heading) + "_" + str(pitch)
                                    image_url = streetview_query + streetview_query_options + "&heading=" + str(heading) + "&pitch=" + str(pitch)
                                    common.save_image_from_url(sign_url(image_url), "data", image_name, real_lat, real_lng)
                                except HTTPError as e:
                                    print(str(heading) + ", " + str(pitch) + " Error: ")
                                    print(sign_url(image_url))
                                    #print(e.read())
            except HTTPError as e:
                print("Error!")
                print()
                print(e.read())               
                            
    print("done")             

    pass


csv_path = 'regions/epcot/park.csv'
coords = []
with open(csv_path, "r") as f:
    for line in f.readlines():
        coords.append([float(line.split(",")[i]) for i in range(len(line.split(",")))])
coords = np.array(coords)

max_lat = "{:.3f}".format(max(coords[:,0]))
max_long = "{:.3f}".format(max(coords[:, 1]))
min_lat = "{:.3f}".format(min(coords[:, 0]))
min_long = "{:.3f}".format(min(coords[:, 1]))

#pdb.set_trace()
# with open("results.json", "w") as g:
#     g.write(json.dumps(photos_search))
# g.close()



if (len(sys.argv) == 2):
    if sys.argv[1] == 'mk':
        print("getting mk")
        get_streetview_in_bounding_box(28.42, -81.586220, 28.422, -81.577347)
    
    if sys.argv[1] == 'epcot':
        print("getting epcot")
        get_streetview_in_bounding_box(28.371, -81.553664, 28.377229, -81.545482)

    if sys.argv[1] == 'dhs':
        print("getting dhs")
        get_streetview_in_bounding_box(28.358, -81.563629, 28.361727, -81.556257)
    
    if sys.argv[1] == 'ak':
        print("getting ak")
        get_streetview_in_bounding_box(28.358, -81.594788, 28.3639, -81.586140)
else:
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    bbox_coords = [min_long, min_lat, max_long, max_lat]
    print(bbox_coords)
    photos_search = flickr.photos.search(bbox = ",".join(bbox_coords), min_upload_date = 1524009600)
    assert(photos_search['stat'] == 'ok')
    photos = photos_search['photos']['photo']
    num=0
    for im in photos:
        common.save_image(im['id'], "Medium", "data")
        print(num)
        num +=1
