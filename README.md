# Expectation-Maximization-Algorithm
EM Algorithm For Image Segmentation On Gray-Scale And RGB Images  -> (Türkçe: RGB ve Gri-Seviyeli Görüntüler İçin EM Algoritmasıyla Segmentasyon İşlemi)

Merhabalar Ben KTÜ Bilgisayar Mühendisliği Yüksek Lisans Öğrencisi Oğuzhan Topal. 
Bu kodda RGB ve Gri-Seviyeli Görüntülerde EM Algoritmasıyla Segmentasyon Yapan Kodu Yazmaya Çalıştım.
Bu benim KTÜ Yüksek Lisans'taki Bilgisayarlı Görme Dersi İçin Yaptığım Arasınav Ödevimdir
Kodumu kullanabilmek için kod dosyası ile aynı dizine bir resim yükleyebilirsiniz. Resmin ismi ne ise koddaki scene.jpg kısmını değiştirmeniz gerekiyor
Ya da imgloc ile istediğiniz dizinden resim yükleyebilirsiniz. 
Kısacası resim yükleme kısmını kendinize göre düzenlemeniz gerekmektedir...


Hello, I'm Oğuzhan Topal, a graduate student in Computer Engineering at Karadeniz Technical University (KTÜ).
The code provided below performs image segmentation using the Expectation-Maximization (EM) algorithm on both gray-scale and RGB images.
This code is developed as part of my midterm assignment for the Computer Vision course at KTÜ.
To use the code, you need to place an image file in the same directory as the code file. 
You can modify the code by replacing the 'scene.jpg' with the name of your image file, or you can specify the image location using the 'imgloc' variable.
Please note that you need to adjust the image loading part according to your needs.

EM Algoritması Nasıl Çalışır:

Kullanılan Parametreler -> Kümelerin Ortalamaları, Kümelerin Varyansları, Ağırlıklandırma Katsayıları

E-Step:

1- Öncelikle her piksel için, her bir kümeye ait olma olasılığını hesaplamamız gerekiyor. Bunu "Gaussian Mixture" Olasılık Fonksiyonu ile yaparız.
2- Bu olasılıkları ağırlıklandırma katsayısı(w) ile çarparak ağırlıklandırırız. Bu katsayılar için başlangıç değerleri 1/Küme Sayısıdır(1/K).
3- Böylece her küme için bütün piksellerin, bu kümelere ait olma olasılıklarını elde etmiş oluruz
4- Daha sonra bu olasılıkları normalize ederiz. Bunun için 3.maddede her küme için bulduğumuz olasılıkları, bu olasılıkların toplamına böleriz.
5- Böylece her bir pikselin her bir kümeye(ya da sınıfa) ait olma olasılığını buluruz.

M-Step:

1- Bu kısımda parametrelerimizi güncelleriz.
2- Kümelerin ortalamaları güncellenir. 
3- Kümelerin varyansları güncellenir.
4- Ağırlıklandırma katsayıları güncellenir.

Likelihood Estimation:

1- Bu kısımda ise her piksel için, normalize edilmiş olasılıkların logaritmalarının toplamını buluruz.
2- Elde ettiğimiz bu değere log_likelihood_estimation deniyor.
3- Eğer bu değer bir önceki adımdaki değer ile belirli bir threshold değerinin altında kalacak şekilde bir farka sahipse ya da başka bir deyişle aralarındaki fark sıfıra yakınsarsa, algoritmayı sonlandırabiliriz.
4- Eğer böyle bir yakınsama olmaz ise, M-Step'te güncellediğimiz parametreleri, E-Step'te kullanarak iterasyonumuza devam ederiz.



How Does the EM Algorithm Work:

Parameters Used -> Cluster Centers (Means), Cluster Variances, Weighting Coefficients

E-Step:

1- First, for each pixel, we need to calculate the probability of belonging to each cluster. We do this using the Gaussian Mixture Probability Function.
2- We weight these probabilities by the weighting coefficient (w). The initial values for these coefficients are 1/Number of Clusters(1/K).
3- This way, we obtain the probabilities of all pixels belonging to these clusters.
4- Then, we normalize these probabilities by dividing them by the sum of probabilities for each cluster.
5- Thus, we obtain the probabilities of each pixel belonging to each cluster (or class).

M-Step:

1- In this step, we update our parameters.
2- We update the cluster centers (means).
3- We update the cluster variances.
4- We update the weighting coefficients.

Likelihood Estimation:

1- In this step, for each pixel, we calculate the sum of logarithms of normalized probabilities.
2- This value is called log_likelihood_estimation.
3- If this value has a difference below a certain threshold compared to the value from the previous step, or in other words, if the difference approaches zero, we can terminate the algorithm.
4- If there is no such convergence, we continue our iterations by using the updated parameters from the M-Step in the E-Step.
