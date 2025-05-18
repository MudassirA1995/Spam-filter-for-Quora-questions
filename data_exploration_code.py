#Data Exploration Code 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#Reading the csv file 
df = pd.read_csv(r"C:\Users\Mudassir\Desktop\Edvancer Assignment Submission\Python 2\Assignment 2\train.csv")

df.head(30)

df.info()

