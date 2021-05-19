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

train=[]
file2 = open(r".\logs.txt","w+")
logsList=[]
conf_matrix=[]
y_test_list=[]


def get_train_score():
    return train[0]
def get_test_score():
    return train[1]

def get_conf_matrix():
    return conf_matrix
def get_y_test():
    return y_test_list


def get_loglist():
    return logsList
def start(test_size=0.25):
    
    # Kullanılacak dataset bir dataframe olarak okunuyor.
    data = './adult.csv'
    df = pd.read_csv(data, header=None, sep=',\s')
    # Sutun isimleri dataset içerinden bir diziye eklenmiştir. Okunan datasetin sutunlarının isimlendirilmesi için kullanılacak.
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
       
    # Sutun isimleri okunan dataframe üzerine ekleniyor.
    df.columns = col_names
    print(df.columns)
        
    # Hedef değişken income sütunu dataframe üzerinden silinerek X değişkenine atanıyor.
    X = df.drop(['income'], axis=1)
    print(X)

    # Hedef değişken income sütunu dataframe üzerinden seçilerek y değişkenine atanıyor.
    y = df['income']
    print(y)
    # Arayüz üzerinden görüntülenmesi için y değişkeni logs.txt dosyasına yazılması için logsList listesine ekleniyor.
    logsList.append(y)
    # X ve y'yi eğitim ve test setlerine ayrılıyor.test_size yani test boyutu kullanıcı tarafından girilen değerin yüzdesi alınarak belirleniyor.
    from sklearn.model_selection import train_test_split
    # X ve y X için hem eğitim hemde test ve y için hem eğitim hemde test olarak 4 parçaya ayrılıyor.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    y_test_list.clear();
    y_test_list.append(y_test)

    numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

    logsList.append('X_train boyutu :')
    logsList.append(X_train.shape)

    logsList.append('X_test boyutu :')
    logsList.append(X_test.shape)

    logsList.append(numerical)
    
    # Eksik kategorik değişkenleri en sık değere dayandırın
    for df2 in [X_train, X_test]:
        df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
        df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
        df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True)
    # Arayüz üzerinden görüntülenmesi için y değişkeni logs.txt dosyasına yazılması için logsList listesine ekleniyor.
    logsList.append(df2)
    # category_encoders : Kategorik değişkenleri farklı tekniklerle sayısal olarak kodlamak için bir dizi scikit-öğrenme tarzı dönüştürücü
    import category_encoders as ce
    
    encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
        'race', 'sex', 'native_country'])

    #İsteğe bağlı fit_params parametreleriyle transformatörü X ve y'ye uyar ve X'in dönüştürülmüş bir sürümünü döndürür 
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    cols = X_train.columns
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    # Naive Bayes Modeli Eğitimi
    from sklearn.naive_bayes import GaussianNB
    # Model nesnesi
    gnb = GaussianNB()
    #Eğitim verileri
    gnb.fit(X_train, y_train)
    # Modelin test edilmesi
    y_pred = gnb.predict(X_test)
    print(y_pred)
    # Arayüz üzerinden görüntülenmesi için y değişkeni logs.txt dosyasına yazılması için logsList listesine ekleniyor.
    logsList.append(y_pred)

    # Metriklerin hesaplanması
    from sklearn.metrics import accuracy_score

    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    file2.writelines('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
        
    # Burada, y_test gerçek sınıf etiketleridir ve y_pred, test kümesindeki tahmin edilen sınıf etiketleridir.
    #Şimdi, aşırı uyumu kontrol etmek için tren seti ve test seti doğruluğunu karşılaştırma

    y_pred_train = gnb.predict(X_train)
    y_pred_train
    logsList.append(y_pred_train)
    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
    
    print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

    # Eğitim seti doğruluk puanı ile  test seti doğruluğu birbirine yakın değerse, bu iki değer oldukça karşılaştırılabilir deneri. Yani, aşırı uyum belirtisi olmaz.
    train.append('Training set score: {:.5f}'.format(gnb.score(X_train, y_train)))
    train.append('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
    
        
    print('Test set score: {:.5f}'.format(gnb.score(X_test, y_test)))

    '''
    Karışıklık matrisi, bir sınıflandırma algoritmasının performansını özetlemek için kullanılan bir araçtır.
    Bir kafa karışıklığı matrisi, bize sınıflandırma modeli performansının ve modelin ürettiği hata türlerinin net bir resmini verecektir.
    Gerçek Pozitifler (TP) - Gerçek Pozitifler, bir gözlemin belirli bir sınıfa ait olduğunu ve gözlemin aslında o sınıfa ait olduğunu tahmin ettiğimizde ortaya çıkar.

    Gerçek Negatifler (TN) - Gerçek Negatifler, bir gözlemin belirli bir sınıfa ait olmadığını ve gözlemin aslında o sınıfa ait olmadığını tahmin ettiğimizde ortaya çıkar.

    Yanlış Pozitifler (FP) - Yanlış Pozitifler, bir gözlemin belirli bir sınıfa ait olduğunu, ancak gözlemin aslında o sınıfa ait olmadığını tahmin ettiğimizde ortaya çıkar. 

    Yanlış Negatifler (FN) - Yanlış Negatifler, bir gözlemin belirli bir sınıfa ait olmadığını, ancak gözlemin aslında o sınıfa ait olduğunu tahmin ettiğimizde ortaya çıkar. 
    Bu çok ciddi bir hatadır ve Tip II hatası olarak adlandırılır.
    '''
    from sklearn.metrics import confusion_matrix
    # Karışıklık Matrisini yazdırın ve dört parçaya bölün
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix.append(cm)
    
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

    
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))
    logsList.append(classification_report(y_test, y_pred))
    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    # sınıflandırma doğruluğu

    print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
    logsList.append('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
    classification_error = (FP + FN) / float(TP + TN + FP + FN)

    # sınıflandırma hatası
    print('Classification error : {0:0.4f}'.format(classification_error))
    logsList.append('Classification error : {0:0.4f}'.format(classification_error))
    
    '''
        Precision
        Kesinlik, tahmin edilen tüm olumlu sonuçlardan doğru şekilde tahmin edilen olumlu sonuçların yüzdesi olarak tanımlanabilir. 
        Gerçek pozitiflerin (TP) doğru ve yanlış pozitiflerin toplamına (TP + FP) oranı olarak verilebilir.

    '''
    precision = TP / float(TP + FP)
    print('Precision : {0:0.4f}'.format(precision))
    logsList.append('Precision : {0:0.4f}'.format(precision))

    '''
    Recall

    Geri çağırma, tüm gerçek olumlu sonuçlardan doğru şekilde tahmin edilen olumlu sonuçların yüzdesi olarak tanımlanabilir.
    Gerçek pozitiflerin (TP), gerçek pozitiflerin ve yanlış negatiflerin (TP + FN) toplamına oranı olarak verilebilir.
    Hatırlama aynı zamanda Hassasiyet olarak da adlandırılır.
    '''
    # recall değeri
    recall = TP / float(TP + FN)
    print('Recall or Sensitivity : {0:0.4f}'.format(recall))
    logsList.append('Recall or Sensitivity : {0:0.4f}'.format(recall))
    
    # True Positive Rate değeri
    true_positive_rate = TP / float(TP + FN)
    print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
    logsList.append('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
    
    # False Positive Rate değeri
    false_positive_rate = FP / float(FP + TN)
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
    logsList.append('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
    
    # Specificity değeri
    specificity = TN / (TN + FP)
    print('Specificity : {0:0.4f}'.format(specificity))
    logsList.append('Specificity : {0:0.4f}'.format(specificity))
    # iki sınıfın tahmin edilen ilk 10 olasılığını yazdırın - 0 ve 1
    y_pred_prob = gnb.predict_proba(X_test)[0:10]
    print(y_pred_prob)
    logsList.append(y_pred_prob)
    #38
    # dataframe içerisinde olasılıkları saklayın
    y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
    print(y_pred_prob_df)
    logsList.append(y_pred_prob_df)
    # 1. sınıf için tahmin edilen ilk 10 olasılığı yazdırın - Olasılık> 50K
    gnb.predict_proba(X_test)[0:10, 1]
    # store the predicted probabilities for class 1 - Probability of >50K
    y_pred1 = gnb.predict_proba(X_test)[:, 1]
    y_test_list.append(y_pred1)
    logsList.append(y_pred1)


    print('---------------------------')
    print(y_test_list[0])
    for line in logsList:
        file2.writelines(str(line))
        file2.writelines('\n-----------------\n')
    file2.close

