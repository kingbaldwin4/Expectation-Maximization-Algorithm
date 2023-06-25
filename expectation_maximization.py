import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors


def expectation_maximization_for_gray(image,K,max_iterations):
    # Resmi düzleştir
    image_vector = image.reshape((-1, 1))
    num_pixels = image_vector.shape[0]

    # Başlangıçta her bir pikselin ait olduğu bileşenin ağırlık değeri, 1/K
    w = [1/K] * K
    # Başlangıçta kullanılacak olan bileşenlerin varyansı, resimdeki piksellerin varyansına göre hesaplanır
    sigma = [np.var(image_vector)] * K


    hist, bins = np.histogram(image_vector.flatten(), 256, [0, 256])
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    centers = np.random.choice(bin_centers, size=K).reshape(-1, 1)

    # Expectation-Maximization algoritmasını uygula
    prev_log_likelihood = -np.inf
    for i in range(max_iterations):
    # Expectation Step
        likelihood = np.zeros((num_pixels, K))
        for j in range(K):
            likelihood[:, j] = w[j] * (1 / np.sqrt(2 * np.pi * sigma[j]) *
                                    np.exp(-1/2 * np.sum(np.square(image_vector - centers[j]), axis=1) / sigma[j]))

        # Normalize likelihoods
        sum_likelihood = np.sum(likelihood, axis=1)
        sum_likelihood_expanded = np.expand_dims(sum_likelihood, axis=1)
        likelihood /= sum_likelihood_expanded


        # Maximization Step
        for j in range(K):
           sum_likelihood_j = np.sum(likelihood[:, j])
           w[j] = sum_likelihood_j / num_pixels
           centers[j] = (np.sum(np.tile(likelihood[:, j].reshape(-1, 1), (1, image_vector.shape[1])) * image_vector, axis=0) / sum_likelihood_j)
           sigma[j] = (np.sum(likelihood[:, j] * np.sum(np.square(image_vector - centers[j]), axis=1)) / sum_likelihood_j)

        # Compute log likelihood
        log_likelihood = np.sum(np.log(np.sum(likelihood * np.tile(np.array(w).reshape(1, -1), (num_pixels, 1)), axis=1)))

        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) < 1e-6:
            print("Yakinsama gerçekleşti...")
            print("Döngü şu adimda sona erdi: ",i)
            break

        prev_log_likelihood = log_likelihood

        # Sıfıra çok yakınsayan sigma değerleri kontrol edilir
        if np.min(sigma) < 1e-6:
            sigma = np.maximum(sigma, 1e-6)

    # Etiketleri tahmin et
    labels = np.argmax(likelihood, axis=1)

    # Görüntüleri bileşenlere atama
    segmented_img = np.zeros_like(image_vector)
    for i in range(K):
        segmented_img[labels == i] = centers[i]
    # Görüntüyü yeniden şekillendir
    segmented_img = segmented_img.reshape((image.shape)).astype(np.uint8)

    # Histogramı oluştur
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].hist(image.flatten(), bins=256, color='gray', alpha=0.7)
    axes[0].set_title('Gri-Seviyeli Goruntu Histogrami')
    axes[0].set_xlabel('Piksel Degeri')
    axes[0].set_ylabel('Frekans')
    cluster_counts = [np.sum(labels == i) for i in range(K)]
    axes[1].bar(range(1, K+1), cluster_counts, color=[f'C{i}' for i in range(K)], edgecolor='black', linewidth=1.2)
    axes[1].set_title('Kume Basina Dusen Piksel Sayisi')
    axes[1].set_xlabel('Kume')
    axes[1].set_ylabel('Piksel Sayisi')

    # Tablo oluştur
    centroid_table = pd.DataFrame({'Kume': [f'Kume {i+1}' for i in range(K)],
                                   'Kume Merkez Degeri': [tuple(center) for center in centers],
                                   'Piksel Sayisi': cluster_counts})
    centroid_table.set_index('Kume', inplace=True)
    print('\n' + '-'*50)
    print('Gri-Seviyeli Goruntu İcin EM Algoritmasi Sonuclari')
    print('-'*50)
    print(centroid_table)

    # Görseli göster
    plt.show()

    # Sonuçları göster
    print("Optimal Kume Merkezleri:", centers)
    cv2.imshow('Gri-Seviyeli Goruntu', image)
    cv2.imshow('Kumelenmis Goruntu', segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def expectation_maximization_for_rgb(image,K,max_iterations):
    # Resmi düzleştir
    image_vector = image.reshape((-1, 3))
    num_pixels = image_vector.shape[0]

    # Başlangıçta her bir bileşenin ağırlık değeri, 1/K
    w = [1/K] * K
    '''# Başlangicta kullanilacak olan bileşenlerin merkezleri, [0, 255] araligindan rastgele seçilir
    centers = np.random.choice(np.arange(256), size=(K, 3))'''

    # Başlangıçta kullanılacak olan bileşenlerin merkezleri, resmin RGB histogramından rastgele seçilir
    hist, bins = np.histogram(image_vector, 256, [0, 256])
    centers = np.zeros((K, 3))
    for c in range(3):
        bin_centers = 0.5*(bins[1:] + bins[:-1])
        centers[:, c] = np.random.choice(bin_centers, size=K)

    # Başlangıçta kullanılacak olan bileşenlerin varyansı, resimdeki piksellerin varyansına göre hesaplanır
    sigma = np.tile(np.asarray([np.var(image_vector[:, i]) for i in range(3)]), (K, 1))

    # Expectation-Maximization algoritmasını uygula
    prev_log_likelihood = -np.inf
    for i in range(max_iterations):
        # Expectation Step
        likelihood = np.zeros((num_pixels, K))
        for j in range(K):
            likelihood[:, j] = w[j] * (1 / np.sqrt(np.prod(2 * (np.pi ** 3) * sigma[j])) *
                                    np.exp(-1/2 * np.sum(np.square(image_vector - centers[j]) / sigma[j], axis=1)))  # güncellendi

        # Normalize likelihoods
        sum_likelihood = np.sum(likelihood, axis=1)
        sum_likelihood_expanded = np.expand_dims(sum_likelihood, axis=1)
        likelihood /= sum_likelihood_expanded

        # Maximization Step
        for j in range(K):
            sum_likelihood_j = np.sum(likelihood[:, j])
            w[j] = sum_likelihood_j / num_pixels
            centers[j] = np.sum(np.tile(likelihood[:, j].reshape(-1, 1), (1, image_vector.shape[1])) * image_vector, axis=0) / sum_likelihood_j
            sigma[j] = np.sum(np.square(image_vector - centers[j]) * np.tile(likelihood[:, j].reshape(-1, 1), (1, image_vector.shape[1])), axis=0) / sum_likelihood_j

        # Compute log likelihood
        log_likelihood = np.sum(np.log(np.sum(likelihood * np.tile(np.array(w).reshape(1, -1), (num_pixels, 1)), axis=1)))

        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) < 1e-6:
            print("Yakinsama gerçekleşti...")
            print("Döngü şu adimda sona erdi: ",i)
            break

        prev_log_likelihood = log_likelihood

        # Sıfıra çok yakınsayan sigma değerleri kontrol edilir
        if np.min(sigma) < 1e-6:
            sigma = np.maximum(sigma, 1e-6)

    # Etiketleri tahmin et
    labels = np.argmax(likelihood, axis=1)

    # Görüntüleri bileşenlere atama
    segmented_img = np.zeros_like(image_vector)
    for i in range(K):
        segmented_img[labels == i] = centers[i]

    # Görüntüyü yeniden şekillendir
    segmented_img = segmented_img.reshape((image.shape)).astype(np.uint8)

    # Histogramı oluştur
    fig, ax = plt.subplots()
    cluster_counts = [np.sum(labels == i) for i in range(K)]
    #ax.bar(range(1, K+1), cluster_counts, color=[f'C{i}' for i in range(K)], edgecolor='black', linewidth=1.2)
    cluster_colors = [(center[2] / 255, center[1] / 255, center[0] / 255) for center in centers]
    ax.bar(range(1, K+1), cluster_counts, color=cluster_colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Kume')
    ax.set_ylabel('Piksel Sayisi')
    ax.set_title('Kume Basina Dusen Piksel Sayisi')

    # Tablo oluştur
    centroid_table = pd.DataFrame({'Kume': [f'Kume {i+1}' for i in range(K)],
                                   'Kume Merkez Degeri': [tuple(center) for center in centers],
                                   'Piksel Sayisi': cluster_counts})
    centroid_table.set_index('Kume', inplace=True)
    print('\n' + '-'*50)
    print('RGB Goruntu İcin EM Algoritmasi Sonuclari')
    print('-'*50)
    print(centroid_table)

    # Görseli göster
    plt.show()

    # Sonuçları göster
    print("Optimal Kume Merkezleri:", centers)
    cv2.imshow('Orjinal Goruntu', image)
    cv2.imshow('Kumelenmis Goruntu', segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('scene.jpg')
K = int(input("Algoritmada Kullanilacak Küme Sayisini Giriniz: "))
iteration = int(input("Algoritmanin Kaç Kez İterasyon Gerçekleştireceğini Giriniz: "))
print("Gri-Seviye Goruntu İcin EM Algoritmasi --> 1")
print("RGB Goruntu İcin EM Algoritmasi --> 2")
choose = int(input("Hangi Fonksiyonu Calistiracaksiniz ? \n"))
if choose == 1:
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    expectation_maximization_for_gray(image,K,iteration)
    exit()
elif choose == 2:
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    expectation_maximization_for_rgb(image,K,iteration)
    exit()
else:
    print("Yanlis Secim...")
    exit()