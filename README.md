![Grafana Ekran Görüntüsü](https://github.com/ynsemrearslan/python_naive_bayes_model/blob/main/ss.png?raw=true)

# Python ile Naïve Bayes Sınıflandırıcı

Bu projede, Bir kişinin yılda 50.000'den fazla kazanıp kazanmadığını tahmin etmek için bir Navie Bayes Sınıflandırıcı kullanıyorum.

## Kurulum
```python
#Gerekliliklerin yüklenmesi
pip install -r requirements.txt
```
```python
#Gereklilikler yüklendikten sonra projenin çalışırılması.
python model
```

## 1. Naïve Bayes algoritması hakkında bilgi

Makine öğreniminde, Naïve Bayes sınıflandırması, sınıflandırma görevi için basit ve güçlü bir algoritmadır. Naïve Bayes sınıflandırması, özellikler arasında güçlü bağımsızlık varsayımı ile Bayes'in teoremini uygulamaya dayanmaktadır. Naïve Bayes sınıflandırması, Doğal Dil İşleme gibi metinsel veri analizi için kullandığımızda iyi sonuçlar verir.

Naïve Bayes modelleri, basit Bayes veya bağımsız Bayes olarak da bilinir. Tüm bu isimler, sınıflandırıcının karar kuralında Bayes'in teoreminin uygulanmasına atıfta bulunur. Naïve Bayes sınıflandırıcısı, Bayes'in teoremini pratikte uygular. Bu sınıflandırıcı, Bayes teoreminin gücünü makine öğrenimine taşır. 
## Naïve Bayes algoritması nerelerde kullanılır?
* Spam filtering
* Text classification
* Sentiment analysis
* Recommender systems

#### Değişken türleri

Bu bölümde, veri setini kategorik ve sayısal değişkenlere ayırıyorum. Veri setinde kategorik ve sayısal değişkenlerin bir karışımı vardır. Kategorik değişkenlerin veri türü nesnesi vardır. Sayısal değişkenler int64 veri türüne sahiptir.

##### Kategorik değişkenlerin özeti

 * 9 kategorik değişken vardır.
 * Kategorik değişkenler çalışma sınıfı, eğitim, medeni_durum, meslek, ilişki, ırk, cinsiyet, yerel_ ülke ve gelir ile verilmektedir.
 * gelir hedef değişkendir.

### Nümerik verilerin özeti
* 6 sayısal değişken vardır.
* Bunlar yaş, fnlwgt, eğitim_sayısı, sermaye_geri, sermaye_kaybı ve saat_başına_hafta ile verilir.
* Tüm sayısal değişkenler ayrı veri tipindedir.

### Veri setini eğitim ve test seti olarak bölünmesi

```python
# X ve y'yi eğitim ve test setlerine ayırın
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

```
## Model eğitimi

```python
# eğitim setinde bir Gaussian Naive Bayes sınıflandırıcısı eğitin
from sklearn.naive_bayes import GaussianNB

# modeli örneklemek 
gnb = GaussianNB()

# modeli eğitme
gnb.fit(X_train, y_train)
```
## Sonuçları tahmin etmek

```python
y_pred = gnb.predict(X_test)

y_pred
```

## Doğruluk puanını kontrol etmek

```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```

#### Tren seti ve test seti doğruluğunu karşılaştırın

Şimdi, aşırı uyumu kontrol etmek için tren seti ve test seti doğruluğunu karşılaştırma.

```python
y_pred_train = gnb.predict(X_train)

y_pred_train
```

```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```
## Karışıklık matrisi

Karışıklık matrisi, bir sınıflandırma algoritmasının performansını özetlemek için kullanılan bir araçtır. Bir kafa karışıklığı matrisi, bize sınıflandırma modeli performansının ve modelin ürettiği hata türlerinin net bir resmini verecektir. Bize her kategoriye göre ayrılmış doğru ve yanlış tahminlerin bir özetini verir. Özet, tablo biçiminde temsil edilir.

Bir sınıflandırma modeli performansını değerlendirirken dört tür sonuç mümkündür. Bu dört sonuç aşağıda açıklanmıştır: -

Gerçek Pozitifler (TP) - Gerçek Pozitifler, bir gözlemin belirli bir sınıfa ait olduğunu ve gözlemin aslında o sınıfa ait olduğunu tahmin ettiğimizde ortaya çıkar.

Gerçek Negatifler (TN) - Gerçek Negatifler, bir gözlemin belirli bir sınıfa ait olmadığını ve gözlemin aslında o sınıfa ait olmadığını tahmin ettiğimizde ortaya çıkar.

Yanlış Pozitifler (FP) - Yanlış Pozitifler, bir gözlemin belirli bir sınıfa ait olduğunu, ancak gözlemin aslında o sınıfa ait olmadığını tahmin ettiğimizde ortaya çıkar. Bu tür bir hataya Tip I hatası denir.

Yanlış Negatifler (FN) - Yanlış Negatifler, bir gözlemin belirli bir sınıfa ait olmadığını, ancak gözlemin aslında o sınıfa ait olduğunu tahmin ettiğimizde ortaya çıkar. Bu çok ciddi bir hatadır ve Tip II hatası olarak adlandırılır.

Bu dört sonuç, aşağıda verilen bir karışıklık matrisinde özetlenmiştir.

## Sınıflandırma ölçütleri

Sınıflandırma raporu, sınıflandırma modeli performansını değerlendirmenin başka bir yoludur. Model için hassasiyet, geri çağırma, f1 ve destek puanlarını görüntüler.

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

#### Precision

Kesinlik, tahmin edilen tüm olumlu sonuçlardan doğru şekilde tahmin edilen olumlu sonuçların yüzdesi olarak tanımlanabilir. Gerçek pozitiflerin (TP) doğru ve yanlış pozitiflerin toplamına (TP + FP) oranı olarak verilebilir.

Dolayısıyla, Kesinlik, doğru tahmin edilen olumlu sonucun oranını tanımlar. Negatif sınıftan çok pozitif sınıfla ilgilenir.

Matematiksel olarak, hassasiyet TP'nin (TP + FP) 'ye oranı olarak tanımlanabilir.

#### Recall
Geri çağırma, tüm gerçek olumlu sonuçlardan doğru şekilde tahmin edilen olumlu sonuçların yüzdesi olarak tanımlanabilir. Gerçek pozitiflerin (TP), gerçek pozitiflerin ve yanlış negatiflerin (TP + FN) toplamına oranı olarak verilebilir. Hatırlama aynı zamanda Hassasiyet olarak da adlandırılır.

Geri çağırma, doğru tahmin edilen gerçek pozitiflerin oranını tanımlar.

Matematiksel olarak hatırlama, TP'nin (TP + FN) 'ye oranı olarak verilebilir.

#### f1-score

f1-score, hassasiyet ve geri çağırmanın ağırlıklı harmonik ortalamasıdır. Olası en iyi f1-score 1.0 ve en kötüsü 0.0 olacaktır. f1-score, hassasiyet ve geri çağırmanın harmonik ortalamasıdır. Dolayısıyla, hesaplamalarına hassasiyet ve geri çağırma ekledikleri için f1-score her zaman doğruluk ölçümlerinden daha düşüktür. F1-score ağırlıklı ortalaması, global doğruluğu değil, sınıflandırıcı modellerini karşılaştırmak için kullanılmalıdır.
## Sonuçlar

* Bu projede, bir kişinin yılda 50.000'den fazla kazanıp kazanmadığını tahmin etmek için Gaussian Naïve Bayes Sınıflandırıcı modeli oluşturuyorum. Model, 0.8083 olarak bulunan model doğruluğunun gösterdiği gibi çok iyi bir performans vermektedir.
* Eğitim seti doğruluk puanı test versinin boyutuna göre 0.8067 iken test seti doğruluğu 0.8083 ile 0.8012 arasında değişmektedir. Bu iki değer oldukça karşılaştırılabilir. Yani, aşırı uyum belirtisi yok.
* Dolayısıyla, Gaussian Naïve Bayes sınıflandırıcı modelimizin sınıf etiketlerini tahmin etmede çok iyi bir iş çıkardığı sonucuna varabiliriz.