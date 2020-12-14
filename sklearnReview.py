#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:24:39 2020

@author: ryanfinegan
"""

import pandas as pd 
from sklearn.datasets import load_boston #boston dataset
from sklearn.linear_model import LinearRegression #linear regression for boston dataset
from sklearn.cluster import KMeans #unsupervised data (no labels / targets)
from sklearn.datasets import load_iris, load_wine #iris and wine
import matplotlib.pyplot as plt #for my graphs for the second question

def boston():
    boston = load_boston() #loading data from sklearn library
    model = LinearRegression() #calling the type of model
    features,label = boston.data, boston.target #using sklearn to split data into features and labels
    model.fit(features,label) #fit model to training data
    df = pd.DataFrame([boston.feature_names,model.coef_]).T #coefficients to see each factors strength
    #wanted to do lasso regression but wasn't sure if that truly answered the question
    return df.sort_values(1,ascending = False) #sorting by the strongest impact each factor has on value 

def clustering():
    iris,wine = load_iris(),load_wine() #load in iris, wine dataset
    inertia,inertia1 = [],[] #empty list to add distance
    for k in list(range(1,15)): #loop through number of clusters
        model = KMeans(n_clusters=k).fit(iris.data) #loop through all K for iris
        model1 = KMeans(n_clusters=k).fit(wine.data) #loop through all K for wine
        inertia.append(model.inertia_) #append inertias to list
        inertia1.append(model1.inertia_) #append inertias to list (sum of squared error in a cluster)
    return inertia,inertia1 #returning both inertias to plot for elbow
    
def plot_elbow(error):
    plt.plot(list(range(1,15)), error) #looping through same 15 clusters
    plt.title("Elbow") #simple title
    plt.ylabel("Inertia") #errors should be y
    plt.xlabel("Clusters") #clusters should be x
    return plt.show() #return the plot

if __name__ == '__main__':
    print(boston()) #printing the boston coefficients from highest to lowest
    inertia,inertia1 = clustering()[0],clustering()[1] #finding the errors of the iris and wine respectively
    # These two graphs show that approximately 3 is in fact where it levels off, 
    print('\nIris Elbow Heuristic')
    plot_elbow(inertia) #plotting iris
    print('Wine Elbow Heuristic')
    plot_elbow(inertia1) #plotting wine
