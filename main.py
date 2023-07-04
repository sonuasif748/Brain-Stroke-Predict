import joblib
import pandas as pd
import numpy as np
import sys
from fastapi import FastAPI
from numpy import percentile
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from pydantic import BaseModel
import gradio as gr


def greet(age,work_type_Never_worked, heart_disease_1, avg_glucose_level,
          ever_married_Yes,hypertension_1,work_type_Private):
    nb = joblib.load('Model')
    # work_type
    if work_type_Never_worked == "Yes":
        work_type_Never_worked = 0
    elif work_type_Never_worked == "No":
        work_type_Never_worked = 1

    # heart disease
    if heart_disease_1 == "Yes":
        heart_disease_1 = 1
    elif heart_disease_1 == "No":
        heart_disease_1 = 0

    # avg_glucose_level
    # avg_glucose_level = float(input("What is your average glucose level( if you don't know, type 100): "))

    # ever_married_yes
    if ever_married_Yes == "Yes":
        ever_married_Yes = 1
    elif ever_married_Yes == "No":
        ever_married_Yes = 0

    # hypertension_1
    if hypertension_1 == "Yes":
        hypertension_1 = 1
    elif hypertension_1 == "No":
        hypertension_1 = 0

    # work_type_private
    if work_type_Private == "Yes":
        work_type_Private = 1
    elif work_type_Private == "No":
        work_type_Private = 0
    user_input = [[age, work_type_Never_worked,
                   heart_disease_1, avg_glucose_level, ever_married_Yes,
                   hypertension_1, work_type_Private]]
    print(user_input)
    pred_user_output = nb.predict(user_input)
    pred_prob_user_output = nb.predict_proba(user_input)
    result = np.round((100 * pred_prob_user_output), 1)
    return "There is a", result[0][1], "% chance that the user will get a stroke."

def gradio_view():
    demo = gr.Interface(fn=greet, inputs=[
            gr.Number(label="How old are you?"),
            gr.Radio(choices = ["Yes", "No"], value=[1,0], label="Have you ever worked?"),
            gr.Radio(choices = ["Yes", "No"], value=[1,0], label="Do you have any heart disease?"),
            gr.Number(label="What is your average glucose level", info="if you don't know, type 100"),
            gr.Radio(choices = ["Yes", "No"], value=[1,0], label="Have you ever been married?"),
            gr.Radio(choices = ["Yes", "No"], value=[1,0], label="Do you have hypertension?"),
            gr.Radio(choices = ["Yes", "No"], value=[1,0], label="Do you work in private sector?"),
                    ], outputs="text")
    demo.launch()

# gradio_view()

app = FastAPI()

class InputData(BaseModel):
    age:int
    work_type_Never_worked:str
    heart_disease_1:str
    avg_glucose_level:float
    ever_married_Yes:str
    hypertension_1:str
    work_type_Private:str


@app.post("/api/predict")
def predict_brain_stroke(reqdata:InputData):
    nb = joblib.load('Model')

    #work_type
    if reqdata.work_type_Never_worked == "Y":
        reqdata.work_type_Never_worked = 0
    elif reqdata.work_type_Never_worked == "N":
        reqdata.work_type_Never_worked = 1

    # heart disease
    if reqdata.heart_disease_1 == "Y":
        reqdata.heart_disease_1 = 1
    elif reqdata.heart_disease_1 == "N":
        reqdata.heart_disease_1 = 0

    # avg_glucose_level
    # avg_glucose_level = float(input("What is your average glucose level( if you don't know, type 100): "))

    # ever_married_yes
    if reqdata.ever_married_Yes == "Y":
        reqdata.ever_married_Yes = 1
    elif reqdata.ever_married_Yes == "N":
        reqdata.ever_married_Yes = 0

    # hypertension_1
    if reqdata.hypertension_1 == "Y":
        reqdata.hypertension_1 = 1
    elif reqdata.hypertension_1 == "N":
        reqdata.hypertension_1 = 0
    else:
        sys.exit("\nINVALID INPUt, PLEASE CHOOSE EITHER Y OR N\n")

    # work_type_private
    if reqdata.work_type_Private == "Y":
        reqdata.work_type_Private = 1
    elif reqdata.work_type_Private == "N":
        reqdata.work_type_Private = 0
    else:
        print("\nINVALID INPUt, PLEASE CHOOSE EITHER Y OR N")
    user_input = [[reqdata.age, reqdata.work_type_Never_worked,
                   reqdata.heart_disease_1, reqdata.avg_glucose_level, reqdata.ever_married_Yes,
                   reqdata.hypertension_1, reqdata.work_type_Private]]
    pred_user_output = nb.predict(user_input)
    pred_prob_user_output = nb.predict_proba(user_input)
    result = np.round((100 * pred_prob_user_output), 1)
    resdata = {"message":result[0][1]}
    return resdata
    # print(pred_user_output)
    # print("\n")
    # print("The results are out:")
    # print("=====================")
    # print("There is a", result[0][1], "% chance that the user will get a stroke.")
    # return "done"

#creating a function to remove outliers
def remove_outliers(data):
    #calculate interquartile range
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25
    #calculate lower and upper limits
    low_lim = q25 - (1.5* iqr)
    up_lim = q75 + (1.5 * iqr)
    #identify and remove outliers
    outliers = []
    for x in data:
        if x < low_lim:
            x = low_lim
            outliers.append(x)
        elif x > up_lim:
            x = up_lim
            outliers.append(x)
        else:
            outliers.append(x)
    return outliers

#creating a function to choose best features
def forward_selection(data, target,significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-
        set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target,
            sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features




@app.post("/api/preprocess")
def pre_process():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    #replace missing values in bmi column using using mean() imputation
    df["bmi"].fillna(df["bmi"].mean(),inplace=True)
    #drop rows with gender=other
    df.drop(df.index[df["gender"]=="Other"], inplace=True)
    #replace "children" values with "never_worked"
    df["work_type"] = df["work_type"].replace(["children"],"Never_worked")
    #rename the residence_type for consistency
    df.rename(columns = {"Residence_type": "residence_type"},inplace=True)
    #change age column into integer from float
    df["age"] = df["age"].astype("int")
    #define categorical variables
    cols=["stroke","gender","hypertension","heart_disease","ever_married",
    "work_type", "residence_type","smoking_status"]
    df[cols] = df[cols].astype("category")
    #drop id column
    df.drop(["id"], axis="columns", inplace=True)
        #removing outliers
    df["bmi"] = remove_outliers(df['bmi'])
    df["avg_glucose_level"] = remove_outliers(df["avg_glucose_level"])
    #get dummies and identify predictor and outcome variables
    predictors = df.drop(columns = ["stroke"])
    outcome = "stroke"
    X = pd.get_dummies(predictors, drop_first=True)
    y = df[outcome]
    #model building #split validation
    ##selecting best features
    # forward_selection(X,y)
    X_for_selected=X[['age', 'work_type_Never_worked',
    'heart_disease_1', 'avg_glucose_level', 'ever_married_Yes',
    'hypertension_1', 'work_type_Private']]
    #splitting the new dataset with best selected features
    trainx, validx, trainy, validy=train_test_split(X_for_selected, y, test_size=0.30,random_state=1)
    #run naive Bayes
    nb = GaussianNB()
    nb.fit(trainx,trainy)
    #predict class membership
    prediction_train_nb=nb.predict(validx)
    #predict probabilities
    pred_train_prob_nb = nb.predict_proba(validx)
    joblib.dump(nb, 'Model')
    return "Model Created successfully"
