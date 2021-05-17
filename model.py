import datetime
import sqlite3
import os
import functools
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
from tkinter import *
from tkinter.ttk import *


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization


import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




class LogWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
        i=0
        file1 = open("./logs.txt","r+")
        self.title("Logs")
        self.geometry("600x300")
        self.lb = Listbox(self)
        for line in file1.readlines():
            self.lb.insert(i,line)
            i=i+1
            print(line)
        self.lb.pack(fill=tk.BOTH, expand=1)

class HotMapWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
      

        figure = Figure(figsize=(5, 4), dpi=100)
        plot = figure.add_subplot(1, 1, 1)

        plot.plot(0.5, 0.3, color="red", marker="o", linestyle="")

        x = [ 0.1, 0.2, 0.3 ]
        y = [ -0.1, -0.2, -0.3 ]
        plot.plot(x, y, color="blue", marker="x", linestyle="")
        
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        
class PredictedWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__()
      

        figure = Figure(figsize=(5, 4), dpi=100)
        plot = figure.add_subplot(1, 1, 1)

        plot.plot(0.5, 0.3, color="red", marker="o", linestyle="")

        x = [ 0.1, 0.2, 0.3 ]
        y = [ -0.1, -0.2, -0.3 ]
        plot.plot(x, y, color="blue", marker="x", linestyle="")
        
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().grid(row=0, column=0)
class PredictModel(tk.Toplevel):
    def __init__(self, master):
        super().__init__()

        


class Model(tk.Tk):
    def __init__(self):
        
        
        super().__init__()

        self.title("Naïve Bayes Classifier in Python")
        self.geometry("500x300")
        self.resizable(False, False)

        self.standard_font = (None, 13)

        self.menubar = tk.Menu(self, bg="lightgrey", fg="black")

        self.log_menu = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.log_menu.add_command(label="View Logs", command=self.show_log_window, accelerator="Ctrl+L")


        self.file_menu = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.file_menu.add_command(label="Close", command=self.close, accelerator="Ctrl+W")
        

        self.mathplot = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.mathplot.add_command(label="HotMap", command=self.show_graph_hotmap_window, accelerator="Ctrl+H")
        self.mathplot.add_command(label="Predicted", command=self.show_graph_predicted_window, accelerator="Ctrl+P")

        self.predict_model = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.predict_model.add_command(label="Predict", command=self.show_predict_window, accelerator="Ctrl+P")

        self.about_menu = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.about_menu.add_command(label="About the model ", command=self.show_log_window, accelerator="Ctrl+L")

        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.menubar.add_cascade(label="Logs", menu=self.log_menu)
        self.menubar.add_cascade(label="Graphs", menu=self.mathplot)
        self.menubar.add_cascade(label="Predict", menu=self.predict_model)
        self.menubar.add_cascade(label="About", menu=self.about_menu)


        self.configure(menu=self.menubar)

        self.main_frame = tk.Frame(self, width=500, height=300, bg="lightgrey")

        self.task_name_label = tk.Label(self.main_frame, text="Veri kümesinin yüzde kaçı test verisi olarak kullanılsın.", bg="lightgrey", fg="black", font=self.standard_font)
        self.task_name_entry = tk.Entry(self.main_frame, bg="white", fg="black", font=self.standard_font)
        self.start_button = tk.Button(self.main_frame, text="Modeli Eğit",command=self.start, bg="lightgrey", fg="black",font=self.standard_font)
        self.pb1 = Progressbar(self, orient=HORIZONTAL,length=100, mode='indeterminate')
        self.pb1.pack(expand=True)
        
        self.train_label_text = tk.StringVar(self.main_frame)
        self.train_label_text.set(" ")
        self.train_label_text = tk.Label(self.main_frame,text='Train sonucu bekleniyor..',bg="lightgrey", fg="black", font=(None, 16))
        
        
        self.test_label_text = tk.StringVar(self.main_frame)
        self.test_label_text.set(" ")
        self.test_label_text = tk.Label(self.main_frame,text='Test sonucu bekleniyor..',bg="lightgrey", fg="black", font=(None, 16))

        self.main_frame.pack(fill=tk.BOTH, expand=1)

        

        self.task_name_label.pack(fill=tk.X, pady=15)
        self.pb1.pack(fill=tk.X, padx=120, pady=(0,20))
        self.task_name_entry.pack(fill=tk.X, padx=50, pady=(0,20))
        self.train_label_text.pack(fill=tk.X ,pady=15)
        self.test_label_text.pack(fill=tk.X ,pady=15)
        self.start_button.pack(fill=tk.X, padx=50)
        self.bind("<Control-l>", self.show_log_window)
        self.bind("<Control-h>",self.show_graph_hotmap_window)
        self.bind("<Control-p>",self.show_graph_predicted_window)

        

    def show_graph_hotmap_window(self,event=None):
        HotMapWindow(self)

    def show_log_window(self, event=None):
        LogWindow(self)
    def show_graph_predicted_window(self, event=None):
        PredictedWindow(self)

    def show_predict_window(self, event=None):
        PredictModel(self)

    def close(self,event=None):
        exit()
    
 
 
 
    def start(self):
        file2 = open(r".\logs.txt","w+")
        logsList=[]
        if not self.task_name_entry.get():
            msg.showerror("No Task", "Please enter a task name")
            return

        self.start_button.config(state=tk.DISABLED)
        self.task_name_entry.configure(state="disabled")
        self.start_button.configure(text="Model eğitimi başladı")
        #1--------------------------
        data = './adult.csv'
        df = pd.read_csv(data, header=None, sep=',\s')
        #2--------------------------
        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        df.columns = col_names
        print(df.columns)
        
        self.update_idletasks()
        self.pb1['value'] += 20
        #3--------------------------
        categorical = [var for var in df.columns if df[var].dtype=='O']
        #4--------------------------
        df['workclass'].replace('?', np.NaN, inplace=True)
        #5--------------------------
        df['occupation'].replace('?', np.NaN, inplace=True)
        #6--------------------------
        df['native_country'].replace('?', np.NaN, inplace=True)

        #7--------------------------
        numerical = [var for var in df.columns if df[var].dtype!='O']
        #8--------------------------
        X = df.drop(['income'], axis=1)

        y = df['income']
        print(y)
        logsList.append(y)
        #9--------------------------
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int(self.task_name_entry.get())/100, random_state = 0)

        #10--------------------------
        numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

        numerical
        logsList.append(numerical)
        #11--------------------------
        numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

        numerical
        logsList.append(numerical)
        self.update_idletasks()
        self.pb1['value'] += 20
        #12--------------------------
        #X_train[categorical].isnull().mean()
        #13--------------------------

       # for col in categorical:
       #     if X_train[col].isnull().mean()>0:
       #        print(col, (X_train[col].isnull().mean()))
        
        #14--------------------------
        for df2 in [X_train, X_test]:
            df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
            df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
            df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)

        #14--------------------------
        import category_encoders as ce
        #15--------------------------
        encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

        X_train = encoder.fit_transform(X_train)

        X_test = encoder.transform(X_test)

        cols = X_train.columns
    
        #16--------------------------
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        self.update_idletasks()
        self.pb1['value'] += 20
       
        #17--------------------------
        X_train = pd.DataFrame(X_train, columns=[cols])
        #18--------------------------
        X_test = pd.DataFrame(X_test, columns=[cols])
        #18--------------------------
        from sklearn.naive_bayes import GaussianNB
        #19--------------------------
        gnb = GaussianNB()
        #20--------------------------
        gnb.fit(X_train, y_train)
        #21--------------------------
        y_pred = gnb.predict(X_test)
        
        print(y_pred)
        logsList.append(y_pred)

        #22--------------------------
        from sklearn.metrics import accuracy_score

        print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
        file2.writelines('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
        #23--------------------------
        y_pred_train = gnb.predict(X_train)

        y_pred_train
        logsList.append(y_pred_train)
        
        #24--------------------------
        print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
        #25--------------------------

        print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
        
        self.train_label_text.config(text='Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
        self.test_label_text.config(text='Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
        self.update_idletasks()

        print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

        #26--------------------------
        null_accuracy = (7407/(7407+2362))
        logsList.append('Null accuracy score: {0:0.4f}'. format(null_accuracy))

        print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

        #27--------------------------
        # Print the Confusion Matrix and slice it into four pieces

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)

        print('Confusion matrix\n\n', cm)
        logsList.append('Confusion matrix :{cm}')

        print('\nTrue Positives(TP) = ', cm[0,0])
        logsList.append('True Positives(TP) = {0:0.4f}'.format(cm[0,0]))
        print('\nTrue Negatives(TN) = ', cm[1,1])
        logsList.append('True Negatives(TN) = {0:0.4f}'.format(cm[1,1]))
        print('\nFalse Positives(FP) = ', cm[0,1])
        logsList.append('False Positives(FP) = {0:0.4f}'.format(cm[0,1]))
        print('\nFalse Negatives(FN) = ', cm[1,0])
        logsList.append('False Negatives(FN) = {0:0.4f}'.format(cm[1,0]))

        #28--------------------------
        from sklearn.metrics import classification_report

        print(classification_report(y_test, y_pred))
        logsList.append(classification_report(y_test, y_pred))
        #29--------------------------
        TP = cm[0,0]
        TN = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        self.update_idletasks()
        self.pb1['value'] += 20
        #30--------------------------

        classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

        print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
        logsList.append('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
        #31--------------------------
        classification_error = (FP + FN) / float(TP + TN + FP + FN)

        print('Classification error : {0:0.4f}'.format(classification_error))
        logsList.append('Classification error : {0:0.4f}'.format(classification_error))
        #32--------------------------
        # print precision score
        precision = TP / float(TP + FP)
        print('Precision : {0:0.4f}'.format(precision))
        logsList.append('Precision : {0:0.4f}'.format(precision))
        #33--------------------------
        recall = TP / float(TP + FN)
        print('Recall or Sensitivity : {0:0.4f}'.format(recall))
        logsList.append('Recall or Sensitivity : {0:0.4f}'.format(recall))
        #34--------------------------
        true_positive_rate = TP / float(TP + FN)
        print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
        logsList.append('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
        #35--------------------------

        false_positive_rate = FP / float(FP + TN)
        print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
        logsList.append('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
        #36--------------------------
        specificity = TN / (TN + FP)
        print('Specificity : {0:0.4f}'.format(specificity))
        logsList.append('Specificity : {0:0.4f}'.format(specificity))
        #37--------------------------
        y_pred_prob = gnb.predict_proba(X_test)[0:10]
        print(y_pred_prob)
        logsList.append(y_pred_prob)
        #38--------------------------
        y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
        print(y_pred_prob_df)
        logsList.append(y_pred_prob_df)
        #39--------------------------
        gnb.predict_proba(X_test)[0:10, 1]

        #40--------------------------
        # store the predicted probabilities for class 1 - Probability of >50K
        y_pred1 = gnb.predict_proba(X_test)[:, 1]
        #41--------------------------
        #42--------------------------
        for line in logsList:
            file2.writelines(str(line))
            file2.writelines('\n-------------------------------------------\n')

        file2.close
        #43--------------------------
        self.update_idletasks()
        self.pb1['value'] += 20
        print('*************************************************')

        #44--------------------------
if __name__ == "__main__":
    model = Model()
    model.mainloop()

