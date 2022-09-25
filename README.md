# Ready4Floods 
Read4Floods is a website that uses machine learning models to deliver chances of floods based on a user's location and time queries.

Technologies Stack 
1) Python
2) Flask
3) Keras
4) Weather API

# Floods in India
![Alt Text](https://media.giphy.com/media/ANbD1CCdA3iI8/giphy.gif)

Flooding occurs when an extreme volume of water is carried by rivers, creeks and many other geographical features into areas where the water cannot be drained adequately. Often during times of heavy rainfall, drainage systems in residential areas are not adequate, or unchecked civil development severely impedes the functionality of an otherwise acceptable drainage system. Floods cause extremely large numbers of fatalities in every country, but due to India's extremely high population density and often under-enforced development standards, a large amount of damages and many deaths which could be otherwise avoided, are allowed to happen. India witnesses flood due to excessive rain which then results in overflow of rivers, lakes and dams, which adds to cause large amounts of damage to people's lives and property. In the past, India has witnessed many of the largest, most catastrophic floods, causing irreparable damage to people's livelihood, property, and crucial infrastructure.

# Our Hack
![Alt Text](https://media.giphy.com/media/5z0cCCGooBQUtejM4v/giphy.gif)

![alt text](https://github.com/TechBusters-CFD/Flood-Prediction-Website/blob/master/ready4floods.jpeg)

We have created a robust and responsive website that uses machine learning to predict the chances and severity of floods in a given region and gives preventive measures to the user based on the result. The machine learning model is a neural network that uses Keras library to predict the chances of flood based on various environmental and geographical factors including but not limited to precipitaion, location, and climatic conditions. The model has been trained on a very detailed and comprehensive data set expanding over the years 1990 to 2015. The model has currently reached 95.6% accuracy on training dataset and we are going to deploy it on the website soon. 

![alt text](https://github.com/TechBusters-CFD/Flood-Prediction-Website/blob/master/response.JPG)

The website will feature a user GUI which takes an input of location and date, which is then processed by Google Maps API to return the accurate location in the Python Flask's backend. This location along with the date is sent to the Machine Learning Model which then returns the severity of the floods, and other parameters. Based on the return data from the Machine Learning Model, the user is displayed with the possibility of a flood and the precautions to take accordingly. If the date is from the past (before 2015), the Machine Learning Model directly returns the details of the actual event.

# Applications
Using this website, we can predict floods that might hit the country in the future. By this method, we can prevent a lot of damages and loss of lives.
