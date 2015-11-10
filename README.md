# Caltrain Rider: A Complete Data Product, IBM Datapalooza 2015

The following material was presented at the [IBM Datapalooza, Nov. 9, 2015](http://www.spark.tc/datapalooza/full-topics/).

Session overview:
*In a finished data product, there are a lot of decisions, both technical and strategic. We'll use the Caltrain Rider app to tell the story of deciding architecture, experimentation with data science, deploying as a mobile app, and how to create a positive data feedback loop. Why did we create this app? We love data. We love trains. Our office is next to a Caltrain Station. Building an intuitive view of the Caltrain systems using information from some of our own sensors (video, audio) combined with information from publicly available sensors (Twitter, Caltrain API) is a fun way to hone our data science skills, while helping to fulfill one of our core values: contributing back to our community.*

The Silicon Valley Data Science Caltrain Rider app provides an intuitive way for Caltrain commuters to understand their personal train schedule. In the future, the application will provide predictions of schedule delays and relevant notifications for riders' routes. 
The purpose of this repo is present several of the key data science components that have been completed for the app. 
Since the data engineering platform and application development environment are already in place, 
we anticipate that a working prediction model will soon be pushed to the Caltrain Rider app. 

## Current Roadmap
- iOS Application
- Published schedule refresh (completed and deployed)
- Realtime schedule pipeline (completed)
- Twitter pipeline (completed)
- Sentiment Analysis (completed)
- Analysis and prediction of delays (in progress)
- Web Service refactor
- Integration testing

#Sentiment Analysis of Caltrain Tweets
Twitter is a popular place for people to vent their frustrations or update their fellow passengers on the current state of public transportation, and as such it is a  valuable source of real-time data on Caltrain service. As a first step to using Twitter activity as a data source for train prediction, we start with a simple question: How do Twitter users currently feel about Caltrain?

This type of problem is well-studied in natural language processing and machine learning, and can be thought of as a classification task: given the content of a document (in our case, a tweet), classify its sentiment as either positive or negative. In the notebook [Sentiment Analysis Of Caltrain Tweets](Sentiment\ Analysis\ of\ Caltrain\ Tweets.ipynb), we walk through an example of this procedure using `scikit-learn` in Python.
