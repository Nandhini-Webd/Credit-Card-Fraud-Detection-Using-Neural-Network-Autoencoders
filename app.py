import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import streamlit as st


def predict(ttype, amt, org_old_bal, org_new_bal, des_old_bal, des_new_bal, model):
    'Code logic to predict using the model. Assumes data is already preprocessed'

    result = []

    # scale and one hot encode before feeding to model
    ttype = ohe[ttype]
    amt = (amt - scaler.loc['Mean', 'amount']) / scaler.loc['Std', 'amount']
    org_old_bal = (org_old_bal - scaler.at['Mean', 'oldbalanceOrg']) / scaler.at['Std', 'oldbalanceOrg']
    org_new_bal = (org_new_bal - scaler.at['Mean', 'newbalanceOrig']) / scaler.at['Std', 'newbalanceOrig']
    des_old_bal = (des_old_bal - scaler.at['Mean', 'oldbalanceDest']) / scaler.at['Std', 'oldbalanceDest']
    des_new_bal = (des_new_bal - scaler.at['Mean', 'newbalanceDest']) / scaler.at['Std', 'newbalanceDest']

    # stack as a numpy array
    temp = np.stack([*ttype, amt, org_old_bal, org_new_bal, des_old_bal, des_new_bal], axis=0)
    temp = temp[np.newaxis]

    # lets make predictions on our samples
    result.append(np.mean(np.square(np.subtract(temp, model.predict(temp))), axis=1))

    # make the boolean predictions
    result.append(result[0] > THRESHOLD[0])

    return result

def main():
    # front end elements of the web page
    html_temp = """
    <body style="background-image: url('image.jpg');">
    <div style ="background-color:tomato;padding:10px">
    <h1 style ="color:black;text-align:center;">Credit Card Fraud Detector</h1>
    </div>
    </body>
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    ttype = st.selectbox('Transaction Type', ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'))
    amt = st.number_input("Transaction Input")
    org_old_bal = st.number_input("Old Balance of Sender")
    org_new_bal = st.number_input("New Balance of Sender")
    des_old_bal = st.number_input("Old Balance of Receipient")
    des_new_bal = st.number_input("New Balance of Receipient")

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = predict(ttype, amt, org_old_bal, org_new_bal, des_old_bal, des_new_bal, model)
        st.success(f'The Transaction is {"fraudulant" if result[1] else "genuine"}! ' +
                   f'Reconstruction Loss: {float(result[0]):.6}')

if __name__ == "__main__":

    NUM_COLS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    # load the model
    model = tf.keras.models.load_model("./saved_model.h5")


    with open("ohe_scaler.pkl", 'rb') as f:
        ohe, scaler = pickle.load(f)

    THRESHOLD = np.array([0.000051])

    main()
