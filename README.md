Abstract
========

Magic Kingdom in Walt Disney World is one of the top 10 most Instagrammed places in the world. Disney Parks hold four out of the ten spots on that list, and having so many publicly available photographs makes it a great area of study for computer vision. Our team was tasked with extracting information about Walt Disney World from photos, completely from scratch. No pre-processed or standardized dataset, no pre-trained models, no labels. As a result, we were able to classify photographs into one of the four Parks and one of twenty-four themed lands, in addition to the approximate location of where each photo was taken.

Introduction
============

There are a plethora of photographs taken in Disney Parks, and while many photographs are iconic such as Cinderella Castle, Walt Disney World spans over 25 thousand acres, a subset of that space being available for visitors to take pictures to share with the world. Using the Google Maps Street View API, we were able to scan Walt Disney World, collecting images throughout the parks.

Although the availability of photgraphs from Google Maps was beneficial to our data collection process, it also was the root of many of our challenges. Many of the datasets we have seen throughout the class were thoroughly pre-processed and filtered to aid in machine learning methods, which although is great for showing consistent results in computer vision technology, it also takes a very long time to create, label, and standardize. In addition, we didn’t particularly want a standardized dataset. A more trivial version of this project would be to create a network that classified the major attractions at Walt Disney World (e.g. Cinderella Castle, Space Mountain, Spaceship Earth), but we wanted to see if a model could learn features of a given park or land from *any* picture taken within the boundaries of the park. This means that somehow, the model had to learn to find common features between a picture of Cinderella Castle with a random store on Main Street, all while having a model that could fit reasonably inside memory.

Even for our team members, this is a difficult task. Although Ryan, a frequent Walt Disney World guest, is able to look at a photo of a street and tell you exactly where it is, the rest of the team would fail to have any idea where many of the photos were taken. If common features existed, they would need to be learned through deep learning.

From there, our team set out to solve three tasks: First, be able to classify each image into one of the four parks at Walt Disney World (Magic Kingdom, Epcot, Hollywood Studios, and Animal Kingdom). This was our baseline task, as a four bin classification seemed to be a reasonable metric for our model.

Second, we wanted to classify each image to one of 24 themed lands. Each park can be partitioned into smaller lands, for example Tomorrowland in Magic Kingdom. This we believed to be a much more difficult model to learn, since there were six times as many labels, and as it followed one sixth of the data to use for each label. That being said, we were still curious about the results, as the model could potentially learn to pick out features that fit a particular land, as the lands all generally shared an aesthetic theme.

Thirdly, inspired by Geoguessr, we hoped to create a model that would attempt to guess the location that a given photograph was taken. We found this idea very interesting, but not super promising. Even with the amount of data we had, the amount of information needed to pinpoint a location in all of Disney World would seem to be very high. That being said, the completion of this task would imply that by virtually walking through the park, our model would be able to map out the park while receiving randomly ordered inputs, stitching together locations without any prior knowledge of the relative location of images.

Related Work
============

The use of CNNs to perform scene recognition is a very recent phenomenon, compared to the age of computer vision as a field of study. AlexNet was largely responsible for this trend.

Method
======

Since one of the larger goals of this project was building this model from the ground up from scratch, we needed to perform all the data collection, processing, training, and testing ourselves. Throughout each step, the team made critical decisions, weighing between the difficulty and usefulness of the task. We knew that we wanted to create a general model, to classify as many aspects of Disney World as possible, but we knew that a model tasked with too general of a problem would fail to converge. Throughout this entire process, we made tradeoffs to keep aligned with our goals (which was easy since we did everything from scratch).

To start, most of our computation was done locally on department machines, but as our scraping and training scaled, we switched over to the compute grid to speed up our process. After a while, due to the high concentration of users on the queue for the compute grid, we switched over to Google Cloud Platform to allow our models to train using multiple GPUs.

Data Collection
---------------

We knew our data needed to be collected to represent Disney World as a whole, not just certain subsets of it. Our original idea, which was later modified, was to scrape the image hosting site Flickr to search within a certain bounding box inside the Disney park. Although Flickr’s API was convenient for this task, we found a few problems. Because Flickr is publically available to post to, many of the pictures, although physically taken at Disney, were not relevant to our search. As an example, many people like to take pictures of their food, so although technically the photograph of a hamburger was taken at Animal Kingdom, that information can hardly be extracted from the image itself. The same applied to photos of people or characters without much scenic information. Although it may have been possible to extract some information from this (Ryan was able to know the particular restaurant where they sold a particular hotdog), we didn’t want to burden our model with this task.

To create a more useful and comprehensive dataset, we used the Google Maps Street View API to do a virtual walk around the park, collecting images at 45 degree heading intervals at every stop position, in addition to looking up and down 30 degrees. We originally looked around in 90 degree intervals instead of 45 degrees. We added the new angles because we realized that having 90 degree intervals meant that images taken from a given location would not have any of the same objects in the field of view. Having images at 45 degree intervals not only increase the amount of data we had, but also allowed for this overlap. All told, we collected 88,656 images of Walt Disney World.

Data Labeling and Processing
----------------------------

Another perk of using the Google Maps Street View API was that we could request that all the images be of the same size (in our case, 600x400). This prevented any need for dealing with multiple size images, which we will discuss later as we discuss our model. This perk allowed us to avoid any pre-processing of the images.

To label each of images to correspond to their park and land, we manually divided Walt Disney World into polygon partitions of Parks and Lands, and wrote a script to place each image into a polygon based on the image’s lattitude and longitude. This allowed for our first two tasks to be completed, and since the Google Maps Street View API was queried using lattitude and longitude data, we had all the labelling needed for our three tasks. See Figure 1 above for an example of the polygons we created.

![Examples of the polygons we created to label our data<span data-label="fig:labeling"></span>](https://i.imgur.com/vTcMIje.jpg){width="linewidth"}

The Model
---------

The Model is a convolutional neural network, consisting of a convolution layer followed by a max pooling layer, a second convolutional and pooling layer, two convolutional layers with a pooling layer, and two fully connected layers to finish it off. Dropout was later applied to the feed forward layers after we started training on multiple epochs to prevent overfitting. Our model is loosely based on Alexnet, but to start off we experimented with different architectures, keeping in mind the constraints of our hardware.

Images were pulled in using a batch size of 120 chosen randomly from our images pulled from Google Maps. The two dropout layers at the end were necessary to counteract reusing images over epochs during training time. Although training time for the model is slow, we hoped that having more and richer features would be able to address the problems discussed above, namely finding patterns and features that would represent the diverse imagery within each of the parks.

Frontend
--------

We also created a quick-and-dirty webpage that allows a user to upload an image and receive feedback about which park and themed land the model believes the photo was taken in, example output of which is shown in Figure 2. It will be available at <http://vision.ryanhecht.net/> until our Google Cloud Platform credit runs out.

![Example output of the interactive webpage for an image of Hollywood Studios<span data-label="fig:labeling"></span>](https://i.imgur.com/EznH9PP.png){width="linewidth"}

Results
=======

Figure 3 shows our training accuracy results for Parks and Lands by batch. As you can see, we enjoyed increasing accuracy, with Land accuracy reaching roughly 55% by batch 900, and Park accuracy reaching 75% by the same batch.

Our Geoguessr-inspired model originally was just randomly guessing latitude and longitude points. As you can see by the graph on the left hand side of Figure 4, the model quickly converged to a loss of 4, which represents a distance of 4 “latitude longitude” units (if latitude and longitude represented a 2d Cartesian plane). This was unacceptable for our purposes, as Walt Disney World is completely contained within one Lat/Lng square. To fix this, we zeroed our coordinates to be centered around Walt Disney World specifically, and scaled them up by 1000 to facilitate learning. The right-hand graph in Figure 4 shows our new results, with the model’s loss decreasing to around 20 units in our transformed coordinate space, which translates to 0.02 “latitude longitude” units, or 1.3 miles at the latitude of Walt Disney World, through 700 batches. This model is still training as this paper is being written, so we’re hoping for additional improvement.

[t] ![image](https://i.imgur.com/M5enFWA.png){width="0.4linewidth"} ![image](https://i.imgur.com/He6HPc6.png){width="0.4linewidth"}

[t] ![image](https://i.imgur.com/NAbaBej.png){width="0.4linewidth"} ![image](https://i.imgur.com/psukHZk.png){width="0.4linewidth"}

Discussion
----------

A few factors contributed to the less than stellar performance of our model.

Firstly, a nontrivial amount of the images we pulled from Google Street View did not contain recognizable features of the location. For example, take the following image:

![image](https://i.imgur.com/Z8PAeef.jpg){width="0.5linewidth"}

As you can see, the contents of this image is almost entirely nondescript foliage. If we were to do this project again, we would likely use a pre-existing scene classifier to discount such uninteresting images from our training dataset.

While Google Street View was a fantastic resource to pull thousands of training images, this information is not free. The static Street View API’s pricing scheme as of 2019 is such that downloading 1000 images costs roughly $7.00. Multiply that by the 88,656 images we collected, and you’ll find that we spend over $600.00 on Google Maps scraping. This was accomplished by burning through the free $300 of trial GCP credit of two of our team members’ Google accounts, plus the trial credit of a new Google Account created specifically for this project (shh, don’t tell Google, this is against their Terms of Service).

Another bottleneck we ran into was computational power dedicated to training our model. We began training on the CS department’s computing cluster, but since many other students were also completing final projects, our jobs would run much slower than we would have liked (occasionally, a batch of 120 images took upwards of five minutes to train). We eventually moved to Google Cloud Platform, but even then, we could not successfully increase our quota to utilize more than one GPU. As a result, we were bottlenecked with how much VRAM we had to work with, and we could only train one model at a time. Significant amounts of time and energy went into setting up the GCP Compute Engine instances for training and the prediction webserver, as well as linking the filesystems between them through GCP Cloud Storage.

Additionally, we were not feasibly able to tinker with the network architecture and hyperparameters as much as we would have liked, since changing the model would mean restarting the training process, and wasting computation time. Since we had over 88,000 images and utilized a batch size of 120, we required 720 batches just for out model to have seen each image once, and since we did make a few changes here and there, we weren’t able to ever train a model over more than 900 batches.

Using Our Code
==============

Install a virtual environment using the dependencies in `requirements.txt`.

Scraping
--------

First, ensure that the “api_key” and “secret” fields of the “google_maps” section of the `config.json` are filled out.

Then, run `python3 scraper.py <mk | epcot | dhs | ak>` to scrape images from Street View in the park you specified. Images will be saved to the `data` folder.

Labeling
--------

To label all images in `data`, run `python3 labeler.py`

Training
--------

To train the park or land models, run `python3 model/model.py -locale <park | land>`

To train the geo model, run `python3 model/geomodel.py`

Predicting
----------

To run the prediction webserver, run `sudo python3 web.py -land <path/to/model> -park <path/to/model> -geo <path/to/model>`, where the model paths begin in the `model` directory.

Using our data/models
---------------------

Our data can be found on the public Google Cloud Platform storage bucket `gs://rpcvisionimages`.

Our park, land, and geo models can be found at `gs://rpcparkmodels`, `gs://rpclandmodels`, and `gs://rpcgeomodels`, respectively.

Conclusion
==========

Image classification is the simplest example at the cross of deep learning and computer vision, and oftentimes the datasets and tests involved in this field involve computers performing tasks that humans can do very well, for example text recognition. The purpose of our project is to explore whether a deep neural net has the capability to pull out subtler traits, seeing if it can find patterns across all the aspects of a large piece of land. Although as expected, our model does not perform as well as simpler tasks, it did show us that this type of task is possible. Traditional methods of general location classification generally use geolocation to cheat their way out of tasks, but we have shown that it is most definitely feasible to perform this task from images as well.

Appendix {#appendix .unnumbered}
========

Team contributions {#team-contributions .unnumbered}
------------------

Please describe in one paragraph per team member what each of you contributed to the project.

Ryan Hecht

:   Ryan worked on labeling each image that was scraped, by writing software that would use the associated location data to map it to the corresponding Park and Land. In addition, he led the effort to move our dataset from Flickr to Google Maps, allowing us to get a wider variety of data. Much of his work with the model involved integrating it with Google Cloud Platform, taking our model and adjusting hyperparameters to work with our image set after he set up our Cloud account, culminating in the web interface to query the model to predict arbitrary images in Walt Disney World.

Peter Hahn

:   Peter experimented with the architecture of the model, balancing the time and memory cost of the model without compromising results too much. Although the final design resembles Alexnet, many changes were made both before and after using Alexnet’s basic architecture to see what would work with the hardware. Peter also made sure that the transition between the data scraping and training was done properly, making sure the label structure worked with the model and writing the code that batched the data for training and testing purposes.

Cristian Luna

:   Cristian worked on the initial data scraping portion of the project, collecting images from the Flickr for our classification purposes. Although we didn’t end up using the Flickr data specifically, he tranfered most of the code to use for our usage of Google Maps. Cristian worked on tuning the model for our task, making sure that the model could accept input for both Parks and Land data, as well as writing code to test our model. In addition, Cristian gave input into the architecture of our model.
