
from model import get_loglist, start
import sys, os
sys.path.append(os.path.abspath("../"))
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as msg
from tkinter import *
from tkinter.ttk import *

from model import get_y_test
from sklearn.metrics import roc_curve

import numpy as np
from numpy.core.fromnumeric import var # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Logların gösterildiği açılan pencere loglar model eğitimi sırasında logs.txt dosyasına yazılıp oradan okuma işlemleri yapılıyor.
class LogWindow(tk.Toplevel):


    def __init__(self, master):
        super().__init__()
        i=0
        self.title("Logs")
        self.geometry("600x300")
        self.lb = Listbox(self)
        for line in get_loglist():
            self.lb.insert(i,line)
            i=i+1
        self.lb.pack(fill=tk.BOTH, expand=1)

# Sıcaklık haritasının gösterildiği açılan pencere
      
# Eğitim durumu grafik halinde gösteren açılır pencere
class PredictedWindow(tk.Toplevel):
    def __init__(self, master,matrix):
        super().__init__()
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class PredictModel(tk.Toplevel):
    def __init__(self, master):
        super().__init__()

# Main sınıfımız model eğitimi ve testi bu sınıfta gerçekleşiyor
class Model(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Naïve Bayes ile Sınıflandırma")
        #Ekran boyutunun piksel olarak verilmesi
        self.geometry("500x300")
        self.resizable(False, False)
        self.standard_font = (None, 13)

        # Üst menü tanımlamaları başlangıcı
        self.menubar = tk.Menu(self, bg="lightgrey", fg="black")

        self.log_menu = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.log_menu.add_command(label="View Logs", command=self.show_log_window, accelerator="Ctrl+L")

        self.file_menu = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.file_menu.add_command(label="Close", command=self.close, accelerator="Ctrl+W")
        

        self.mathplot = tk.Menu(self.menubar, tearoff=0, bg="lightgrey", fg="black")
        self.mathplot.add_command(label="HotMap", command=self.show_graph_hotmap_window, accelerator="Ctrl+H")
        self.mathplot.add_command(label="Predicted", command=self.show_graph_predicted_window, accelerator="Ctrl+P")
        self.mathplot.add_command(label="ROC", command=self.show_predict_window)

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
        # Üst menü tanımlamaları bitişi

        self.main_frame = tk.Frame(self, width=500, height=300, bg="lightgrey")

        # Veri kümesinin yüzde olarak kullanıcı tarafından alınan değere göre bölünmesi sağlanıyor.
        self.task_name_label = tk.Label(self.main_frame, text="Veri kümesinin yüzde kaçı test verisi olarak kullanılsın.", bg="lightgrey", fg="black", font=self.standard_font)
        self.task_name_entry = tk.Entry(self.main_frame, bg="white", fg="black", font=self.standard_font)
        self.start_button = tk.Button(self.main_frame, text="Modeli Eğit",command=self.start, bg="lightgrey", fg="black",font=self.standard_font)

        # Modelin eğitim ilerlemesi bir progress bar yardımıyla gösteriliyor.
        self.pb1 = Progressbar(self, orient=HORIZONTAL,length=100, mode='indeterminate')
        self.pb1.pack(expand=True)
        
        # Eğitim sonucunun gösterildiği label
        self.train_label_text = tk.StringVar(self.main_frame)
        self.train_label_text.set(" ")
        self.train_label_text = tk.Label(self.main_frame,text='Train sonucu bekleniyor..',bg="lightgrey", fg="black", font=(None, 16))
        
        # Test sonucunun gösterildiği label
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
        
    # Üst mendüde yer alan butonların işlemleri için fonksiyonlar
    def show_graph_hotmap_window(self,event=None):
        from model import get_conf_matrix
        cm=get_conf_matrix()[0]
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        classNames = ['Negative','Positive']
        plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        plt.show()
    def show_log_window(self, event=None):
        LogWindow(self)
    def show_graph_predicted_window(self,event=None):
        
        y_test=get_y_test()[0]
        y_pred1=get_y_test()[1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

        plt.figure(figsize=(6,4))

        plt.plot(fpr, tpr, linewidth=2)

        plt.plot([0,1], [0,1], 'k--' )

        plt.rcParams['font.size'] = 12

        plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

        plt.xlabel('False Positive Rate (1 - Specificity)')

        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.show()
    def show_predict_window(self, event=None):
        y_pred1=get_y_test()[1]
        # adjust the font size 
        plt.rcParams['font.size'] = 12
        # plot histogram with 10 bins
        plt.hist(y_pred1, bins = 10)
        # set the title of predicted probabilities
        plt.title('Histogram of predicted probabilities of salaries >50K')
        # set the x-axis limit
        plt.xlim(0,1)
        # set the title
        plt.xlabel('Predicted probabilities of salaries >50K')
        plt.ylabel('Frequency')
        plt.show()
    def close(self,event=None):
        exit()
    def start(self,event=None):
        if not self.task_name_entry.get():
            msg.showerror("No Task", "Please enter a task name")
            return
        start(test_size=int(self.task_name_entry.get())/100)
        from model import get_train_score
        from model import get_test_score
        self.train_label_text.config(text=get_train_score())
        self.test_label_text.config(text=get_test_score())
        self.update_idletasks()      

if __name__ == "__main__":
    model = Model()
    model.mainloop()