import pandas as pd 

#make a data frame for all current CSV fiel
#dor valuescount on lable column
#print the result

hf_frac_test = pd.read_csv('pseudo_labels_HF_frac_test.csv')

# print(hf_frac_test.head())

print(f""" 
       HF Frac Test Predicted Label: {hf_frac_test.predicted_label.value_counts()}
       """)

# len(hf_frac_test[hf_frac_test['predicted_label] == 0])

hf_frac_train = pd.read_csv('pseudo_labels_HF_frac_train.csv')

# print(hf_frac_train.head())

print(f"""
      HF Frac Train Predicted Label: {hf_frac_train.predicted_label.value_counts()}
       """)

hf_frac_valid = pd.read_csv('pseudo_labels_HF_frac_valid.csv')

# print(hf_frac_valid.head())

print(f"""
      HF Frac Valid Predicted Label: {hf_frac_valid.predicted_label.value_counts()}
      """)

hf_nfrac_test = pd.read_csv('pseudo_labels_HF_nfrac_test.csv')

# print(hf_nfrac_test.head())

print(f""" 
       HF NFrac Test Predicted Label: {hf_nfrac_test.predicted_label.value_counts()}
       """)


hf_nfrac_train = pd.read_csv('pseudo_labels_HF_nfrac_train.csv')

# print(hf_nfrac_train.head())

print(f"""
      HF NFrac Train Predicted Label: {hf_nfrac_train.predicted_label.value_counts()}
       """)

hf_nfrac_valid = pd.read_csv('pseudo_labels_HF_nfrac_test.csv')

# print(hf_nfrac_valid.head())

print(f"""
      HF NFrac Valid Predicted Label: {hf_nfrac_valid.predicted_label.value_counts()}
      """)

