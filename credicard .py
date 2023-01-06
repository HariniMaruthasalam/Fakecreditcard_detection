# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:03:59 2023

@author: harin
"""

import numpy as np
import pickle
import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open('C:/Users/harin/PROJECT ML/trained_model (1).sav','rb'))

def fakecreditcard_prediction(input_data):
    
    #changing the input data into numpy array
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)
    pred= loaded_model.predict(id_reshaped)
    print(pred)

    if(pred[0]==0):
        print("orginal")
    else:
        print("fraud")
   
    
def main():
    
    st.title('FAKE CREDIT CARD PREDICTION')
    
    Amount = st.text_input('Amount value')
    Class = st.text_input('Class value')
    
    # Prediction code
    predictionn = ''
    
    if st.button('PRED'):
        predictionn = fakecreditcard_prediction([Amount, Class])
        
    st.success(predictionn)
    
if __name__=='__main__':
    main()