## Caltrain Rider: A Complete Data Product, IBM Datapalooza 2015

The following material was presented at the [IBM Datapalooza, Nov. 9, 2015](http://www.spark.tc/datapalooza/full-topics/).

**Session overview:**
*In a finished data product, there are a lot of decisions, both technical and strategic. We'll use the Caltrain Rider app to tell the story of deciding architecture, experimentation with data science, deploying as a mobile app, and how to create a positive data feedback loop. Why did we create this app? We love data. We love trains. Our office is next to a Caltrain Station. Building an intuitive view of the Caltrain systems using information from some of our own sensors (video, audio) combined with information from publicly available sensors (Twitter, Caltrain API) is a fun way to hone our data science skills, while helping to fulfill one of our core values: contributing back to our community.*

The Silicon Valley Data Science Caltrain Rider app provides an intuitive way for Caltrain commuters to understand their personal train schedule. In the future, the application will provide predictions of schedule delays and relevant notifications for riders' routes. 
The purpose of this repo is present several of the key data science components that have been completed for the app. 
Since the data engineering platform and application development environment are already in place, 
we anticipate that a working prediction model will soon be pushed to the Caltrain Rider app. 

### Project Outline
- iOS Application
- Published schedule refresh (completed and deployed)
- Realtime schedule pipeline (completed)
- Twitter pipeline (completed)
- Sentiment Analysis (completed)
- Analysis and prediction of delays (in progress)
- Web Service refactor
- Integration testing


## Sentiment Analysis of Caltrain Tweets

[Sentiment Analysis of Caltrain Tweets.ipynb](https://github.com/silicon-valley-data-science/datapalooza-2015/blob/master/Sentiment%20Analysis%20of%20Caltrain%20Tweets.ipynb)

Twitter is a popular place for people to vent their frustrations or update their fellow passengers on the current state of public transportation, and as such it is a  valuable source of real-time data on Caltrain service. As a first step to using Twitter activity as a data source for train prediction, we start with a simple question: How do Twitter users currently feel about Caltrain?

This type of problem is well-studied in natural language processing and machine learning, and can be thought of as a classification task: given the content of a document (in our case, a tweet), classify its sentiment as either positive or negative. In the notebook [Sentiment Analysis Of Caltrain Tweets](Sentiment\ Analysis\ of\ Caltrain\ Tweets.ipynb), we walk through an example of this procedure using `scikit-learn` in Python.

## Trainspotting

[trainspotting-demo.ipynb](https://github.com/silicon-valley-data-science/datapalooza-2015/blob/master/trainspotting-demo.ipynb)

This demo will walk you through the creation of an algorithm that uses image processing techniques and heuristics unique to our problem of detecting when a train passes by our office and in which direction it traveling. 
The following steps outline the content of the notebook:

- Load video
- Convert to grayscale
- Smooth the frames
- Compute running average
- Detect changes
- Adjust the difference threshold
- Compare right/left sides of video
- Detect train 
- Detect direction of train

**Technical note**: The trainspotting-demo depends on [motion_detector.py](https://github.com/silicon-valley-data-science/datapalooza-2015/blob/master/motion_detector.py) and the video file 'test.avi'.

## Detecting Catastrophic Delays from Twitter

[caltrain_tweets.ipynb](https://github.com/silicon-valley-data-science/datapalooza-2015/blob/master/caltrain_tweets.ipynb)

Caltrain occasionally experiences system-wide perturbations that make train departures 
unpredictable. 
When this happens, Caltrain shuts down their real-time API.
This occurs during:

- pedestrian strikes
-vehicle strikes
- stalled trains

The frequency and duration of such events are unpredictable. 
In some cases, trains experience long delays due to externalities,
such as:

- major sporting events
- 'heavy ridership'
- busy holiday weekends
- construction

While these are often known in advance, 
their exact effects on train delays are difficult to predict. 
The purpose of this analysis is to detect such catastrophic delays in order shift the operating
regime of our prediction model.
As trains begin to operate normally along the corridor, our model must recover reliably.

We thus examine catastrophic events in historical data for two reasons:

- Exclude them from predictible delays for the purposes of training our model,
- Identify them in real-time so that the prediction model can be shifted to a different regime

For the initial stages, we manually inspected Caltrain's twitter feed.
Tweets are loaded from the data file [caltrain_tweets.json](https://github.com/silicon-valley-data-science/datapalooza-2015/blob/master/caltrain_tweets.json). 
During a 10-day period in October, we identified four major incidents, 
three of which resulted in service interruptions and major delays.
