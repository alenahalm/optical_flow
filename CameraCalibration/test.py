import numpy as np
import cv2 as cv
import glob
import pickle


with open('./dist.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)