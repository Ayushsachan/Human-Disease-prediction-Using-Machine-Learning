from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

A1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite',
'polyuria','family_history','mucoid_sputum', 'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking',
'pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',' Migraine',
'Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E',
'Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia',
'Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo']

A2=[]
for x in range(0,len(A1)):
    A2.append(0)


tst=pd.read_csv("Training.csv")

tst.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,
'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,
'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

print(tst.head())

X= tst[A1]

y = tst[["prognosis"]]
np.ravel(y)
print(y)

tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,
'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,
'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}},inplace=True)

X_test= tr[A1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

def SupportVectorMachine():
    
    from sklearn.svm import SVC
    
    clf1 = SVC()
    clf1 = clf1.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred = clf1.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(A1)):
        for z in psymptoms:
            if(z==A1[k]):
                A2[k]=1

    inputtest = [A2]
    predict = clf1.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

def LogisticRegression():
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    clf2 = LogisticRegression()
    clf2 = clf2.fit(X,np.ravel(y))
    

    prediction = clf2.predict(x_test)
    y_pred = clf2.predict(X)
    from sklearn.metrics import classification_report
    classification_report(y_test,predictions)
    from sklearn.metrics import accuracy_score
   
    accuracy_score = clf2.score(X, y_pred)
    print(accuracy_score)
    

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(A1)):
        for z in psymptoms:
            if(z==A1[k]):
                A2[k]=1

    inputtest = [A2]
    predict = clf2.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf3 = RandomForestClassifier()
    clf3 = clf3.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(A1)):
        for z in psymptoms:
            if(z==A1[k]):
                A2[k]=1

    inputtest = [A2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

def DecisionTree():

    from sklearn import tree

    clf4 = tree.DecisionTreeClassifier()   
    clf4 = clf4.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred =  clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(A1)):
        for z in psymptoms:
            if(z==A1[k]):
                A2[k]=1

    inputtest = [A2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t4.delete("1.0", END)
        t4.insert(END, disease[a])
    else:
        t4.delete("1.0", END)
        t4.insert(END, "Not Found")    
       

def NaiveBayes():
    
    from sklearn.naive_bayes import GaussianNB
    
    clf5 = GaussianNB()
    clf5 = clf5.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred = clf5.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(A1)):
        for z in psymptoms:
            if(z==A1[k]):
                A2[k]=1

    inputtest = [A2]
    predict = clf5.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t5.delete("1.0", END)
        t5.insert(END, disease[a])
    else:
        t5.delete("1.0", END)
        t5.insert(END, "Not Found")
        

root = Tk()
root.configure(background='white')
root.title("HUMAN DISEASE PREDICTOR")

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

Heading = Label(root, justify=LEFT, text="Disease Prediction using Machine Learning", fg="Black", bg="white")
Heading.config(font=("Times New Roman", 30))
Heading.grid(row=1, column=0, columnspan=10, padx=250, sticky= W)

OPTIONS = sorted(A1)

NameLb = Label(root, text="Name of the Patient", fg="ivory", bg="dark blue")
NameLb.grid(row=4, column=0, pady=10, padx= 2, sticky=W)
NameEn = Entry(root, width = 22)
NameEn.grid(row=4, column=1)

GenderLb = Label(root, text="Gender", fg="white", bg="dark blue")
GenderLb.grid(row=4, column=2, pady=10, padx= 2, sticky=W)
Gender = ("Select","Male", "Female", "Transgender")
GenderEn = ttk.Combobox(root, values=Gender, state='readonly')
GenderEn.grid(row=4, column=2, pady=10, padx= 20)         
GenderEn.current(0)

AgeLb = Label(root, text="Age", fg="white", bg="dark blue")
AgeLb.grid(row=5, column=0, pady=10,padx= 2, sticky=W)
Age = ["Select","1 year", "2 year", "3 year", "4 year", "5 year", "6 year", "7 year", "8 year", "9 year", "10 year", "11 year", "12 year", "13 year", "14 year", "15 year", "16 year", "17 year", "18 year", "19 year", "20 year", "21 year", "22 year", "23 year", "24 year", "25 year", "26 year", "27 year", "28 year", "29 year","30 year", "31 year", "32 year", "33 year", "34 year", "35 year", "36 year", "37 year", "38 year", "39 year", "40 year", "41 year", "42 year", "43 year", "44 year", "45 year", "46 year", "47 year", "48 year", "49 year", "50 year", "51 year", "52 year", "53 year", "54 year", "55 year", "56 year", "57 year", "58 year", "59 year", "60 year", "61 year", "62 year", "63 year", "64 year", "65 year", "66 year", "67 year", "68 year", "69 year", "70 year", "71 year", "72 year", "73 year", "74 year", "75 year", "76 year", "77 year", "78 year", "79 year", "80 year", "81 year", "82 year", "83 year", "84 year", "85 year", "86 year", "87 year", "88 year", "89 year", "90 year", "91 year", "92 year", "93 year", "94 year", "95 year", "96 year", "97 year", "98 year", "99 year", "100 year"]
AgeEn = ttk.Combobox(root,values=Age, state='readonly')
AgeEn.grid(row=5, column=1, pady=10, padx= 20)
AgeEn.current(0)

BloodGroupLb = Label(root, text="Blood Group", fg="white", bg="dark blue")
BloodGroupLb.grid(row=5, column=2, pady=10, padx= 2, sticky=W)
BloodGroup = ["Select", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
BloodGroupEn = ttk.Combobox(root, values=BloodGroup, state='readonly')
BloodGroupEn.grid(row=5, column=2, pady=10, padx= 20)
BloodGroupEn.current(0)

S1Lb = Label(root, text="Symptom 1", fg="White", bg="green")
S1Lb.grid(row=10, column=0, pady=10, padx= 2, sticky=W)
S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=10, column=2)

S2Lb = Label(root, text="Symptom 2", fg="White", bg="green")
S2Lb.grid(row=11, column=0, pady=10, padx= 2, sticky=W)
S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=11, column=2)

S3Lb = Label(root, text="Symptom 3", fg="White", bg="green")
S3Lb.grid(row=12, column=0, pady=10, padx= 2, sticky=W)
S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=12, column=2)

S4Lb = Label(root, text="Symptom 4", fg="White", bg="green")
S4Lb.grid(row=13, column=0, pady=10, padx= 2, sticky=W)
S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=13, column=2)

S5Lb = Label(root, text="Symptom 5", fg="White", bg="green")
S5Lb.grid(row=14, column=0, pady=10, padx= 2, sticky=W)
S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=14, column=2)

svmLb = Label(root, text="Support Vector Machine", fg="white", bg="red")
svmLb.grid(row=17, column=0, pady=10, padx= 2, sticky=W)
svm = Button(root, text="Support Vector Machine", command=SupportVectorMachine,bg="Blue",fg="White")
svm.grid(row=10, column=3,padx=10)

lgLb = Label(root, text="Logistic Regression", fg="white", bg="red")
lgLb.grid(row=19, column=0, pady=10, padx= 2, sticky=W)
lg = Button(root, text="Logistic Regression", command=LogisticRegression,bg="blue",fg="white")
lg.grid(row=11, column=3,padx=10)

ranfLb = Label(root, text="Random Forest", fg="white", bg="red")
ranfLb.grid(row=21, column=0, pady=10, padx= 2, sticky=W)
ranf = Button(root, text="Random forest", command=randomforest,bg="Blue",fg="white")
ranf.grid(row=12, column=3,padx=10)

destreeLb = Label(root, text="Decision Tree", fg="white", bg="red")
destreeLb.grid(row=23, column=0, pady=10,padx= 2, sticky=W)
destree = Button(root, text="Decision Tree", command=DecisionTree,bg="Blue",fg="White")
destree.grid(row=13, column=3,padx=10)

nbLb = Label(root, text="Naive Bayes", fg="white", bg="red")
nbLb.grid(row=25, column=0, pady=10, padx= 2, sticky=W)
nb = Button(root, text="Naive Bayes", command=NaiveBayes,bg="Blue",fg="white")
nb.grid(row=14, column=3,padx=10)


t1 = Text(root, height=1, width=40,bg="light blue",fg="black")
t1.grid(row=17, column=2 , padx=10)

t2 = Text(root, height=1, width=40,bg="light blue",fg="black")
t2.grid(row=19, column=2 , padx=10)

t3 = Text(root, height=1, width=40,bg="light blue",fg="black")
t3.grid(row=21, column=2, padx=10)

t4 = Text(root, height=1, width=40,bg="light blue",fg="black")
t4.grid(row=23, column=2 , padx=10)

t5 = Text(root, height=1, width=40,bg="light blue",fg="black")
t5.grid(row=25, column=2 , padx=10)

root.mainloop()
