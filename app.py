import streamlit as st
import pandas as pd
import pickle
import joblib
import re
import pandas as pd
import numpy as np
import re
import string
from string import digits
from sklearn import metrics
import pickle
import time
from sentence_transformers import SentenceTransformer

# Create a Streamlit app
st.title("Gallagher : Text Classification and Excel Processing App")

# File upload for Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])


import base64
from io import BytesIO

def get_binary_file_downloader_link(file_data, file_name, link_text):
    # Write the DataFrame to an in-memory Excel file
    excel_buffer = BytesIO()
    file_data.to_excel(excel_buffer, index=False, engine='xlsxwriter')
    
    # Create a base64-encoded string of the Excel file's contents
    b64 = base64.b64encode(excel_buffer.getvalue()).decode()
    
    # Generate the download link
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}">{link_text}</a>'
    
    return href


def pre_processing(data_frame):

    # Lowercase all characters
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: x.lower())

    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"won\'t", "will not", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"can\'t", "can not", x))

    # general
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"n\'t", " not", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'re", " are", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'s", " is", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'d", " would", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'ll", " will", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'t", " not", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'ve", " have", x))
    data_frame['Claim Description'] = data_frame['Claim Description'].apply(lambda x: re.sub(r"\'m", " am", x))

    # Remove quotes
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: re.sub("'", '', x))



    exclude = set(string.punctuation) # Set of all special characters
    # Remove all the special characters
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: x.translate(remove_digits))


    # remove extra
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: re.sub('[-_.:;\[\]\|,]', '', x))


    # Remove extra spaces
    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: x.strip())

    data_frame['Claim Description']=data_frame['Claim Description'].apply(lambda x: re.sub(" +", " ", x))
    
    return data_frame

step_1_model_path = "output/lr_step_1.pickle"
step_2_model_path = "output/lr_basemodel_step_2.pickle"

step_1_model = pickle.load(open(step_1_model_path, 'rb'))
step_2_model = pickle.load(open(step_2_model_path, 'rb'))
count_vector_step_1 = joblib.load("output/count_vector_step_1.pkl")
count_vector_step_2 = joblib.load("output/count_vector_step_2.pkl")
fewer_class_dict = joblib.load("output/fewer_class_dictionary.pkl")
acc_src_model = joblib.load("output/bert_acc_src.pickle")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



def predict(model_1,model_2,final_dict,query):
    # predict
    
    test_1 =  count_vector_step_1.transform([query])
    y_pred = model_1.predict(test_1)
    if y_pred == 'med':
        test_2 =  count_vector_step_2.transform([query])
        y_pred = model_2.predict(test_2)
    else:
        y_pred = y_pred
        
    if query in final_dict.keys():
        y_pred = final_dict[query]
    else:
        y_pred = y_pred
        
    return y_pred[0]                                                 

if uploaded_file is not None:
    # Read the uploaded Excel file
    excel_data = pd.read_excel(uploaded_file)


    final_result= []
    print('Preprocessing Started')
    test_data = pre_processing(excel_data)
    x_test = test_data['Claim Description']
    print('Prediction Started')
    for query in x_test:
        result = predict(step_1_model,step_2_model,fewer_class_dict,query)
        final_result.append(result)
    excel_data['predicted_coverage_code'] = final_result


    X_bert_enc = model.encode(x_test.values, show_progress_bar=True,)
    accident_source_pred = acc_src_model.predict(X_bert_enc)
    excel_data['predicted_accident_src'] = accident_source_pred


    st.dataframe(excel_data)  # Display the processed data


    link = get_binary_file_downloader_link(excel_data, 'my_processed_file.xlsx', 'Download Processed Data')
    st.markdown(link, unsafe_allow_html=True)


    # Create a new Excel file with the processed data
    output_filename = "processed_data.xlsx"
    excel_data.to_excel(output_filename, index=False)

    # Display a link to download the processed file
    st.markdown(f"Download Processed Data: [Processed Data](data:{output_filename})")



# Add a placeholder for displaying "Done" after processing
if uploaded_file is not None:
    st.write("Done")
