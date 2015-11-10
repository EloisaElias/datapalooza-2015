# datapalooza-2015

The Silicon Valley Data Science Caltrain Rider app provides an intuitive way for Caltrain commuters to understand their personal train schedule. In the future, the application will provide predictions of schedule delays and relevant notifications for riders' routes. 
The purpose of this repo is present several of the key data science components that have been completed for the app. 
Since the data engineering platform and application development environment are already in place, 
we anticipate that a working prediction model will soon be pushed to the Caltrain Rider app. 

#Current Roadmap
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
