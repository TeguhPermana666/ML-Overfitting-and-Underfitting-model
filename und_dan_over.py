"""
underfitting and overfitting digunakan utnuk membuat model lebih akurat

#experiment dengan model yang berbeda:
Sekarang setelah Anda memiliki cara yang andal untuk mengukur akurasi model, 
Anda dapat bereksperimen dengan model alternatif dan melihat mana yang memberikan prediksi terbaik.Tapi apa alternatif yang Anda miliki untuk model?
Anda dapat melihat dalam dokumentasi scikit-learn bahwa model pohon keputusan memiliki banyak opsi 
(lebih dari yang Anda inginkan atau butuhkan untuk waktu yang lama). Pilihan yang paling penting menentukan kedalaman pohon.
Ingat dari pelajaran pertama dalam kursus ini bahwa kedalaman pohon adalah ukuran berapa banyak perpecahan yang dibuatnya
sebelum sampai pada prediksi. Ini adalah pohon yang relatif dangkal

Dalam praktiknya, tidak jarang sebuah pohon memiliki 10 belahan antara tingkat atas (semua rumah) dan sehelai daun.
Saat pohon semakin dalam, kumpulan data terpotong menjadi daun dengan lebih sedikit rumah.
Jika sebuah pohon hanya memiliki 1 split, ia membagi data menjadi 2 kelompok.
Jika setiap kelompok dibagi lagi, kita akan mendapatkan 4 kelompok rumah. Memisahkan masing-masing lagi akan membuat 8 grup.
Jika kita terus menggandakan jumlah grup dengan menambahkan lebih banyak perpecahan di setiap level, kita akan memiliki 210 grup rumah pada saat kita mencapai level 10. Itu 1024 daun.
Ketika kami membagi rumah di antara banyak daun, kami juga memiliki lebih sedikit rumah di setiap daun.
Daun dengan rumah yang sangat sedikit akan membuat prediksi yang cukup mendekati nilai sebenarnya dari rumah tersebut,
tetapi mereka mungkin membuat prediksi yang sangat tidak dapat diandalkan untuk data baru (karena setiap prediksi hanya didasarkan pada beberapa rumah).

Ini adalah fenomena yang disebut overfitting, di mana model cocok dengan data pelatihan hampir sempurna,
tetapi kurang dalam validasi dan data baru lainnya. Di sisi lain, 
jika kita membuat pohon kita sangat dangkal, itu tidak membagi rumah menjadi kelompok yang sangat berbeda.

Pada ekstremnya, jika sebuah pohon membagi rumah menjadi hanya 2 atau 4, 
setiap kelompok masih memiliki berbagai macam rumah. Prediksi yang dihasilkan mungkin jauh untuk sebagian besar rumah,
bahkan dalam data pelatihan (dan validasinya juga akan buruk karena alasan yang sama). Ketika sebuah model gagal menangkap perbedaan dan pola penting dalam data, 
sehingga kinerjanya buruk bahkan dalam data pelatihan, itu disebut underfitting.

contoh:
Ada beberapa alternatif untuk mengontrol kedalaman pohon, dan banyak yang memungkinkan beberapa rute 
melalui pohon memiliki kedalaman yang lebih besar daripada rute lainnya. Tetapi argumen max_leaf_nodes memberikan cara yang
sangat masuk akal untuk mengontrol overfitting vs underfitting. Semakin banyak daun yang kita izinkan untuk dibuat model,
semakin banyak kita berpindah dari area underfitting pada grafik di atas ke area overfitting.
Kita dapat menggunakan fungsi utilitas untuk membantu membandingkan skor MAE dari nilai yang berbeda untuk max_leaf_nodes:
"""
#membandingkan keakuratan model in sample scores dengan train_split_model

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
#in sample scores

def get_mae(max_leaf_nodess,train_x,val_x,train_y,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodess,random_state=0)
    #train
    model=model.fit(train_x, train_y)
    predcdts_val=model.predict(val_x)
    mae=mean_absolute_error(val_y, predcdts_val)
    return mae


#train_test_split model

#load data
import pandas as pd
file_path="intro to ml\melb_data.csv"
model_data_melbroune=pd.read_csv(file_path)
#filtered missing value
filltered_model_data_melbroune=model_data_melbroune.dropna(axis=0)
#chose target predictions adn fitur
y=filltered_model_data_melbroune.Price
fitur=['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']
X=filltered_model_data_melbroune[fitur]
#split data model
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(X,y,random_state=0)
# #define model
# model_data_tssm=DecisionTreeRegressor(random_state=0)
# #train
# model_data_tssm=model_data_tssm.fit(train_x,train_y)
# #validate
# val_predicds=model_data_tssm.predict(val_x)
# mae=mean_absolute_error(val_y, val_predicds)



#gunakan sebuah perulangan untuk mengetahui runtime perubahanya

#compare MAE with differing values of max_leaf_nodes
for max_leaf_nodess in [5,50,500,5000]:
    my_MAE=get_mae(max_leaf_nodess, train_x, val_x, train_y, val_y)
    print("My_leaf_nodes {}\t\t{}".format(max_leaf_nodess,my_MAE))
"""
Kesimpulan
Inilah kesimpulannya: Model dapat terjadi sebuah peristiwa:

=>Overfitting: menangkap pola palsu yang tidak akan terulang di masa mendatang, yang mengarah ke prediksi yang kurang akurat, atau
=>Underfitting: gagal menangkap pola yang relevan, sekali lagi mengarah ke prediksi yang kurang akurat.
Kami menggunakan data validasi, yang tidak digunakan dalam pelatihan model, untuk mengukur akurasi model kandidat. 
Ini memungkinkan kami mencoba banyak model kandidat dan mempertahankan yang terbaik.
"""