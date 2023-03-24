# Designing Tattoos with a Neural Network

## Introduction

  Tattoos are personal, permanent, and pretty hard to design. While it’s exciting to engage in the intimate process of working with a tattoo artist to help design a tattoo, that process can take a long time and cost a lot of money. It’s common to get inspiration from images of tattoos online, but what if you don’t want a design that’s trending? Where do you go if you want to design something that is completely unique? Consider consulting a neural network. 
	Current art generators can produce tattoo recommendations, though they may look unrealistic and intimidating due to an emphasis on artistic value over reasonable tattoo recommendations. There do exist, however, several implementations of tattoo-specific generative models. These models are built on generative adversarial networks (GANs) which produce a collection of unique tattoos after being trained on an input dataset of tattoo images scraped from various websites. 
	We aim to build our own tattoo GAN from scratch, using FastAI to create a dataset of tattoo images. While this sounds lofty, we want to pursue it in order to best learn the inner workings of a generative neural network. If after 2 weeks we have not built a model that successfully creates unique tattoos, we will begin working with various existing tattoo GANs, creating our own datasets, and comparing the model’s output.
	This is an important area of research because 36% of young adults in the United States have tattoos (History of Tattoos, 2023). This is a large number, but consider the many more people who might be considering getting a tattoo but haven’t yet and might benefit from unique inspiration. This is also a difficult area of research because of the controversies around using generative models for art. Many artists have expressed frustration with these models for mimicking their work, raising questions about the ethics of producing images and about what unique artwork actually is.
  Tattoo creation by Neural Network is an interesting problem to address, especially as it has barely been done before. There are many models  which have been trained using different artwork, and can create an original product. However, models have been rarely trained solely on images of tattoos, either cleaned pictures of tattoos or not-including skin and other background components. Our research found only two people who attempted such a project, but their models have not been widely used, if at all beyond personal purposes. Moreover, tattoo generation is complicated due to the various ethical issues involved, as outlined in the ethics section.
Therefore, we would like to try and create a new model that will enable better results in the long run. This requires research on different types of Neural Networks (GANs, Stable Diffusion for example) and creating a new model that will effectively comply with our demands.
  To make the generated tattoo images more realistic, we will augment traditional image generation neural networks with parameters like color, simplicity of the style, location, and size. 
The technical challenges of this project will be to learn how to implement a working neural network. Creating a model from scratch will be difficult to learn as students with minimal exposure to neural networks, however will be vital in strengthening our understanding of deep learning concepts. Our team aims to be able to train a basic neural network model which will use AI-scraped data. We expect that scraping such data may prove to be difficult, and may face challenges such as size, quality, and quantity when it comes to finding tattoo images. In other words, using fastAI, we will be able to find images of tattoos but are concerned about where and how to crop the resulting images and if enough images of tattoos exist publicly. 
  This model will potentially be capable of recommending images from the generated database. The team will cite and research other GAN models when producing their own original model. It is unclear whether a tangible output will be feasible for our project.
  This project will build a tattoo generation model based on original architecture. The generator will be built toward the capability to produce tattoos (image) based upon user input (text) while producing uniquely designed art which attempts to avoid plagiarism of other artistic works. The model will provide one option per user request; if time permits, will be extended to more.
  
  
## Related Works

There are several current generative tattoo models that utilize neural nets to produce unique tattoo images.

Tattoodo, an online platform developed by Vuksic, utilized neural networks to recognize and classify different tattoo styles, offering users personalized tattoo feed based on preferred artists, styles (such as tribal, water color, or traditional), and motifs (such as flora, swords, or dragons). Developers used a deep learning network called Caffe along with Nvidia’s Deep Learning GPU Training System to train the neural network. DALL-E2, a neural network created by OpenAI that “turns natural human language into realistic photos and art” (Hood, 2022) has been used by people to design tattoos as well. 

In his 2020 Medium post Vasily Betin describes how his love for tattoos and technology led him to create his own tattoo generation model. He outlines how he created his dataset of images of tattoos by scraping Instagram and Pinterest, cleaned the data, trained the network on low then high resolution images, and then continued to upscale and perfect the model. After generating over 2,000 images, Vasily chose his favorites and actually got one tattooed on his arm. The model is released on RunwayML with a public Github Repo. 


## Ethical Sweep

  Looking at the ethical issues that might be raised from this project there are a few things that we can think about. First, however, is looking at whether we should be doing this to begin with. Since it is a class project that is interesting to us, we believe that we should be doing this. Nevertheless, in real life, there should be a deeper thought and restraints/precautions for this project–acknowledgement of the datasets, sources, and artists, as well as age restriction, and the words that can be used for our Tattoo generator. Additionally, we need to consider the diversity of our team. We are not representative of all skin colors, we come from cultures and families that don’t support tattoos, and none of us have tattoos. 
  There are a few ethical issues with our project which are outlaid here:
Copyrights→ As we are dealing with artistic pieces, which are generated by Tattoo artists who usually have copyrights over those pieces of art.We need to make sure that the data we use in our dataset does not conflict with copyrights. It is also important to ask how Tattoo artists may feel with such a program, which is taking away part of their living and artistic value.
  Skin color for Data→ Our data needs to be diverse enough for it to include different types of skin color. Black vs white would not be sufficient here, and the data does need to aim for a more diverse set. We would say that a guideline of about 100 shades of skin color is the minimum (though we don’t promise to deliver that in this project).
  Encouraging young people to get a tattoo→ Such a program may encourage young people to have a tattoo at an earlier age. Just having such a program may give this incentive, but also the reduced cost due to having a final tattoo proposal instead of having the tattoo artists draw one.
Indigenous culture→ Some cultures have a higher meaning of having tattoos than just body paintings. In these cultures, having a tattoo can have a meaning of maturity, tribal connection, stage of life, age, genealogical connection and many others. For this project, we would say that it is ok to use such data in our dataset. We would put a disclaimer regarding such an issue, and will try to acknowledge tribal connection and information when adequate.
  Offensive symbols→ By allowing users to specify what type of tattoo they want generated, there is potential for the creation (accidental or not) of offensive words and symbols. The dataset the model is trained on may contain tattoos with offensive symbols, but for the scope of this project we do not have the capacity to screen our dataset as thoroughly as we would like.
  What words are used→ Since many existing tattoos may contain words of profanity, the data we trained on may lead to tattoo generations with hatred and disrespect. We need to ensure that generated tattoos avoid discriminating words or terms which are offensive to groups. 
  Providing the same tattoo over and over again→ This may be a more personal issue for private users who want to have a distinct tattoo as much as possible. We cannot guarantee that the program will not supply the exact same tattoo, which should also be added a a disclaimer, as well as providing the program with some variation parameter, that will make slight changes between every graphic we provide.
  
  
## Methods

Our methods for this project fall into three major steps. First, we will collect images to create a tattoo dataset. Second, we will create pieces of a GAN from scratch (the discriminator and generator neural networks). Third, we will utilize and modify an existing StyleGAN to generate tattoo images.

We will create our dataset by scraping images from Google Images and using FastAI. 
We aim to collect 500 distinct images, where similar images would not count as distinct. If we are unable to do so we will reduce the goal number in accordance with what we are able to scrape for this project, taking into consideration the limitation of data and especially time and cost. We could also use data augmentation if we don’t have enough data: cropping images, changing backgrounds and skin tones, zooming in and zooming out, etc. 

The challenges we may face when collecting our data is that the scraped images will have varying backgrounds which may make it difficult to isolate the data required. For example, tattoo images may appear on body parts or art on a plain white background. We plan to address this problem by attempting to use a helper function or program which can remove the backgrounds of tattoos, and placing them on homogeneous backgrounds.

For our model, we will use an existing GAN (such as Pytorch DCGAN,  text-2-image, or StyleGAN). We will first use only the outline of these models and attempt to write our own neural networks that will act as a discriminator and a generator. We will connect the generator and discriminator: formatting input and output, configuring the dataset, and fine-tuning the parameters. Then we will train our network on the images from our dataset and run it. After that, we will fully utilize an existing GAN (likely StyleGAN), train on our images, and create new images from the network. We will look at the results and modify certain attributes such as learning rate, number of epochs, and batch size as needed. 

Our project will focus on generating realistic tattoo images on people’s bodies, within the constraints of the class.


## References
<a id="1">[1]</a>
Betin, Vasily. “Artificially Generated Tattoo.” Artificially Generated Tattoo, Medium, 12 Mar. 2020, https://medium.com/vasily-betin/artificially-generated-tattoo-2d5fbe0f5146. 

<a id="2">[2]</a>
Hood, Lonnie Lee. “Man Gets Tattoo of Art Created by Neural Network.” Man Gets Tattoo of Art Created By Neural Network, Futurism, 10 Apr. 2022, https://futurism.com/the-byte/tattoo-created-by-neural-network. 

<a id="3">[3]</a>
Schweitzer, Annette. “How Ai Can Help You Find the Perfect Tattoo.” Getting Good Ink: How AI Can Help You Find the Perfect Tattoo, NVIDIA Blog, 29 Sept. 2017, https://blogs.nvidia.com/blog/2017/09/29/find-the-perfect-tattoo/. 

<a id="4">[4]</a>
Vuksic, Goran. “Ai and Tattoos: How We Built a Neural Network for Tattoo Style Recognition.” AI and Tattoos: How We Built a Neural Network for Tattoo Style Recognition, Medium, 25 Aug. 2017, https://blog.tattoodo.io/ai-and-tattoos-how-we-built-a-neural-network-for-tattoo-style-recognition-6e641df99a05. 
