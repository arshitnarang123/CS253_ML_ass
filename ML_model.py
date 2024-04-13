import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('train.csv')
#function to find the constituency type(SC, ST or unreserved)
def extract_constituency_type(constituency):
    if "(SC)" in constituency:
        return 1
    elif "(ST)" in constituency:
        return 0
    else:
        return 3
#function to find extra info about candidate(Doctor or Advocate)
def extract_extra_info(name):
    if "Dr." in name:
        return 4
    elif "Adv." in name:
        return 2
    else:
        return 0     
#Creating a new column constituency type
df['Constituency_Type'] = df['Constituency ∇'].apply(extract_constituency_type)
#Creating a new column containing extra info about candidate
df['Extra'] = df['Candidate'].apply(extract_extra_info)
#Creating a new dataframe by dropping non-required columns
selected_columns = [ 'Party', 'Criminal Case','Total Assets','Liabilities','state','Education','Constituency_Type','Extra']
df_selected = df[selected_columns]
#applying one hot encoding on 'state' and 'Party' column
columns_to_encode = ['state', 'Party']
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df_selected[columns_to_encode])
encoded_columns_names = encoder.get_feature_names_out(columns_to_encode)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns_names)
df_selected.drop(columns=columns_to_encode, inplace=True)
df_selected = pd.concat([df_selected, encoded_df], axis=1)
#applying label encoding on 'Education' column
label_encoder1 = LabelEncoder()
df_selected['Education'] = label_encoder1.fit_transform(df_selected['Education'])
#Converting total assets string into numerical value
df_selected['Total Assets'] = df_selected['Total Assets'].astype(str)
# Remove both 'Crore+' and 'Lac+' parts
df_selected['Total Assets'] = df_selected['Total Assets'].str.replace('+', '').str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5').str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
# Convert the column to numeric values
df_selected['Total Assets'] = pd.to_numeric(df_selected['Total Assets'], errors='coerce')
#Converting Liabilities string into numerical value
df_selected['Liabilities'] = df_selected['Liabilities'].astype(str)
# Remove both 'Crore+' and 'Lac+' parts
df_selected['Liabilities'] = df_selected['Liabilities'].str.replace('+', '').str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5').str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
# Convert the column to numeric values
df_selected['Liabilities'] = pd.to_numeric(df_selected['Liabilities'], errors='coerce')
#taking mean of total assets to replace zero values with this value
assets_mean = df_selected['Total Assets'].mean()
df_selected['Total Assets'] = df_selected['Total Assets'].replace(0, assets_mean)

X = df_selected.drop(columns=['Education'])
y = df_selected['Education']


# Step 3: Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=60)

# Traing the model using Bernoulli Naive Bayes
clf = BernoulliNB(alpha=0.40, binarize=0.0,fit_prior=True,class_prior=None)
clf.fit(X, y)

# Step 5: Make Predictions
# y_pred = clf.predict(X_test)

# # Step 6: Evaluate the Model
# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)
test = pd.read_csv('test.csv')
#Creating a new column constituency type
test['Constituency_Type'] = test['Constituency ∇'].apply(extract_constituency_type)
#Creating a new column containing extra info about candidate
test['Extra'] = test['Candidate'].apply(extract_extra_info)
#Dropping non required columns
test = test.drop(columns=['ID', 'Candidate', 'Constituency ∇'])
#Applying one hot encoding on 'state' and 'party' column
encoded_data = encoder.fit_transform(test[columns_to_encode])
encoded_columns_names = encoder.get_feature_names_out(columns_to_encode)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns_names)
test.drop(columns=columns_to_encode, inplace=True)
test = pd.concat([test, encoded_df], axis=1)
#converting total assets string into numerical value
test['Total Assets'] = test['Total Assets'].astype(str)
# Remove both 'Crore+' and 'Lac+' parts
test['Total Assets'] = test['Total Assets'].str.replace('+', '').str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5').str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
# Convert the column to numeric values
test['Total Assets'] = pd.to_numeric(test['Total Assets'], errors='coerce')
# Let's say your DataFrame is df and the column name is 'column_name'
test['Liabilities'] = test['Liabilities'].astype(str)
# Remove both 'Crore+' and 'Lac+' parts
test['Liabilities'] = test['Liabilities'].str.replace('+', '').str.replace(' Crore', 'e+7').str.replace(' Lac', 'e+5').str.replace(' Thou', 'e+3').str.replace(' Hund', 'e+2')
# Convert the column to numeric values
test['Liabilities'] = pd.to_numeric(test['Liabilities'], errors='coerce')

y_pred = clf.predict(test)

predicted_labels = label_encoder1.inverse_transform(y_pred)

# print("Predicted Education Labels:")
# print(predicted_labels)

# Create a DataFrame for the predicted labels and corresponding IDs
df_result = pd.DataFrame({'ID': np.arange(len(y_pred)), 'Education': predicted_labels})

# Save the DataFrame to a CSV file
df_result.to_csv('education_output.csv', index=False)

print("CSV file saved successfully.")
