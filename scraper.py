import numpy as np 
import requests
import os 
import re 
import json
import flickrapi
import pdb

api_key =  u'de2346d37930a57db36e02ccc155673b'
api_secret = u'c4d4b868c3878f89'

pdb.set_trace()
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
bbox_coords = (-81.577874, 28.416726, -81.577897, 28.416738)
photos_search = flickr.photos.search(bbox = bbox_coords)



# for i in range(num_images):