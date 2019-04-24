# common functions
import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb
import urllib.request
import csv


# Save the Flickr image with name `image_name` of size `size` to the folder specified in `path`
# It will be saved as `path`/`image_name`.jpg
# Its lat/lng will be saved in `path`/labels/`image_name`.json
def save_image(image_name, size, path):
    sizes = flickr.photos.getSizes(photo_id=image_name)['sizes']['size']

    medium_size = next(s for s in sizes if s['label'] == size)
    urllib.request.urlretrieve(medium_size['source'], path + "/" + image_name + ".jpg")

    location = flickr.photos.geo.getLocation(photo_id=image_name)['photo']['location']
    lat_lng = {'lat': location['latitude'], 'lng': location['longitude']}
    update_label(path + "/labels/" + image_name + ".json", {'location': lat_lng})

def save_image_from_url(url, path, image_name, lat, lng):
    urllib.request.urlretrieve(url, path + "/" + image_name + ".jpg")
    lat_lng = {'lat': lat, 'lng': lng}
    update_label(path + "/labels/" + image_name + ".json", {'location': lat_lng})

# Update the json in `file` with the values in the dictionary `dict`. File will be created if it doesn't exist,
#   and existing json data will be maintained. If a key in `dict` matches a json key in `file`, it will be overwritten

# TODO: FIX IMPLEMENTATION. THIS IS CURRENTLY BROKEN AND WILL OVERWRITE EVERYTHING IN THE FILE WITH THE PASSED IN DICT
def update_label(file, dict):
    with open(file, 'w+') as lbl_file:
        #print(file)
        if os.stat(file).st_size == 0:
            data = {}
        else:
            #try:
            #data = json.load(lbl_file)
            data = {}
        
        data.update(dict)
        
        json.dump(data, lbl_file)

with open("config.json", 'r') as file:
    config = json.load(file)
api_key =  config['flickr']['api_key']
api_secret = config['flickr']['api_secret']
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    