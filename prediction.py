from joblib import load
import pandas as pd 

model = load(r"D:\Loan Project\model_dir\Loan_model.joblib")

sample_data = pd.DataFrame({
  "Dependents" : ["0"],
  "Education" : ["0"],
  "ApplicantIncome" : [3000],
  "CoapplicantIncome" : [1000],
  "LoanAmount" : [1500],
  "Loan_Amount_Term" : [180],
  "Gender_Male" : ["1"],
  "Married_Yes" : ["0"],
  "Self_Employed_Yes" : ["0"],
  "Credit_History_1.0" :["1"],
  "Property_Area_Semiurban" :["0"],
  "Property_Area_Urban" : ["1"]
})

prediction = model.predict(sample_data)
print(prediction)