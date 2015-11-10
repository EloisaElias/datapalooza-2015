# datapalooza-2015

The current Trains application provides an intuitive way for Caltrain riders to understand their personal train schedule. In the future, the application will provide predictions of schedule delays and relevant notifications for rider's route.

#Purpose 

- Develop engineering and data science skills
- Increase SVDS Visibility
- Build platform for future data science and application development
- Develop platform for testing viability of asset/product ideas
- Onboarding project for new jointers

#Current Roadmap
- iOS Application
- Published schedule refresh (completed and deployed)
- Realtime schedule pipeline (completed)
- Twitter pipeline (completed)
- Sentiment Analysis (completed)
- Analysis and prediction of delays (in progress)
- Web Service refactor
- Integration testing

#Sentiment Analysis of Caltrian Tweets
Twitter is a popular place for people to vent their frustrations or update their fellow passengers on the current state of public transportation, and as such it is a  valuable source of real-time data on Caltrain service. As a first step to using Twitter activity as a data source for train prediction, we start with a simple question: How do Twitter users currently feel about Caltrain?

This type of problem is well-studied in natural language processing and machine learning, and can be thought of as a classification task: given the content of a document (in our case, a tweet), classify its sentiment as either positive or negative. In the notebook [Sentiment Analysis Of Caltrain Tweets](Sentiment\ Analysis\ Of\ Caltrain\ Tweets.ipynb), we walk through an example of this procedure using `scikit-learn` in Python.