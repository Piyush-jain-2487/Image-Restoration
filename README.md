# Image-Restoration

  ## Problem Statement:
In the digital world where sending images over different platform has become a norm. Sometimes to deliver images different platform compress the original image and image can be compromised i.e. quality deterioration.
    
  ## Solution:
We studied research paper to learn about deep learning and started implementing solution to restore deteriorates images. We also created an interface where we can upload images and restored images can be downloaded. We developed solution for image denoising, artifacts removal & super-resolution.

  ## Requirements:
- The system must let users upload images from any device.
- The system must let users select the type of image restoration they want to perform on the uploaded image.
- The system must allow users to view processed images.
- The system must allow users to download processed images.
- The system should keep record of performance for analysis and improvement.
- The system should be able to handle simultaneous user requests.
- The system should convert the uploaded image to required size before processing.
- The system requires a python environment.
- The system may run any CUDA supported Operating System
- The system requires a Graphics Processing Unit (GPU) to work efficiently.
- 8GB Video memory for GPU is required and must be a CUDA enabled GPU.
- CPU must be equivalent to Intel® Core™ i7 5th gen or higher in performance.
- Python, NumPy and PyTorch
- Flask, uwsgi and nginx
- Pillow

## System Architecture:
- Our system consists of multiple subsystems. Each of the subsystems is an Artificial Neural Network(ANN) that basically follows the architecture of a Convolutional Neural Network(CNN). For Image Denoising, we use Multi-level Wavelet CNN(MWCNN) which is based on U-Net Architecture while Image Inpainting uses a Generative Image Inpainting network and Image Super resolution uses Enhanced Super Resolution Generative Adversarial Network(ESRGAN), both of which are based on Generative Adversarial Networks(GAN).
![image](https://user-images.githubusercontent.com/47841108/123908999-078dfe00-d996-11eb-96ba-c307db09bda0.png)


- ### A Neuron Layer in Convolutional Neural Network:
  ![image](https://user-images.githubusercontent.com/47841108/123909475-b92d2f00-d996-11eb-9207-78f13d9b87c0.png)



- ### Convolution Neural Network:
  ![image](https://user-images.githubusercontent.com/47841108/123909822-3193f000-d997-11eb-85ea-718557c6fd45.png)



- ### Generative Adversarial Network
  - GAN architecture consists of two different types of neural networks, a generator network and a discriminator network. The generator takes randomly generated noise as input and tries to estimate the required output while the discriminator is trained on a set of real data to differentiate between real and fake data. If the discriminator is near perfect, then output from the generator is tested and if the error rate of discriminator increases, it means that the output of the generator is very close to the real data set. The generator trains on the output available from the discriminator. The generator solves a regression problem while the discriminator solves a classification problem.
![image](https://user-images.githubusercontent.com/47841108/123910543-36a56f00-d998-11eb-97da-3e105764b981.png)



- ### U-Net:
  - U-Net is named after its “U” shaped Architecture. It first reduces the size of feature maps successively and then rebuilds the feature maps using up-convolutions and the previously available feature maps of the same size. As it reduces the size of the feature maps, each feature map begins to represents a pattern in the original image. Thus, the reduction is performed until these patterns are generalized enough for the requirements. The up-convolutions use the lower sized feature maps as a base to form an output based on the recognized pattern but at the same time using original feature maps of the same size to form a resultant image similar to the original.
![image](https://user-images.githubusercontent.com/47841108/123910677-6f454880-d998-11eb-9588-9e25f13d984f.png)



- ### Multi-Wavelet Convolution Neural Network:
  - MWCNN uses Discrete Wavelet transform (DWT) and convolutions to take advantage of the nature of noisy signals in images. It follows a U-Net architecture where the input image is transformed into wavelet domain; it is then convolved following a U-net architecture wherein it uses DWT to reduce the size of feature maps instead of pooling operations and Inverse discrete Wavelet Transform (IWT) instead of up-convolutions for reconstruction of feature maps.
![image](https://user-images.githubusercontent.com/47841108/123910843-a3b90480-d998-11eb-8392-067ddb269d96.png)



- ### Enhanced Super Resolution Generative Adversarial Network:
  - ESRGAN defines the generator as made up of multiple dense residual blocks. Each dense residual block (RB) in turn consists of multiple residual blocks. Each residual block is made up of multiple convolutional layers with outputs from previous layers as input to each layer. Each output from a residual block is also input to further residual blocks. This way, at each layer, the feature maps constantly depend on previous layers. Thus, the output generated is ensured to be closely related to the input image.
![image](https://user-images.githubusercontent.com/47841108/123911045-e549af80-d998-11eb-8245-6ebb84081ed2.png)

  - ESRGAN also uses a Relativistic Average Discriminator loss function for the discriminator which denotes the relative percentage of "real" or "fake" for each of the generator‟s output image instead of the absolute real or fake labels used by a standard GAN.
![image](https://user-images.githubusercontent.com/47841108/123911178-0f9b6d00-d999-11eb-92b6-772e85913594.png)



- ### Generative Image Impainting:
  - Generative Image Inpainting follows the GAN architecture where it processes the image in two steps, first to produce a coarse estimate for the required image and then builds upon that coarse image to fine tune the details. It uses a special contextual attention layer for the refinement part where it compares the adjacent pixels of the selected area for inpainting with that of the estimated result produce; This results in the images being more in line with the surrounding areas of the patch that was to be inpainted.
![image](https://user-images.githubusercontent.com/47841108/123911826-e4fde400-d999-11eb-8588-7eaab6ee510b.png)
![image](https://user-images.githubusercontent.com/47841108/123911723-c5ff5200-d999-11eb-9085-360d1e436c96.png)



 - ### Web-App Architecture:
    ![image](https://user-images.githubusercontent.com/47841108/123912384-98ff6f00-d99a-11eb-93e9-532756ccb7ec.png)



# Source:
  - Deep Image Prior by Dmitry Ulyanov et al. (https://arxiv.org/pdf/1711.10925.pdf)
  - Enhanced Super-resolution Generative Adversarial Network by Xintao Wang et al. (https://arxiv.org/pdf/1809.00219.pdf)
  - Multi-level Wavelet CNN for Image Restoration by Pengju Liu et al. (https://arxiv.org/abs/1805.07071)
  - Generative Image Inpainting with Contextual Attention by Jiahui Yu et al. (https://arxiv.org/abs/1801.07892)
