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


## Methods


## Results
We trained our GAN separately on two dataset - colorful images and black and white images.

We had better results with the black and white dataset. We saw that with a too low of a learning rate we had pretty bad results, and landed on a learning rate of 0.0005. Decreasing Batch size and increasing epochs number improved the results, to a limit. Running a batch size of 32, 1000 epochs, on the color images dataset had worse results than a batch of 32 with 500 epochs. 1000 epochs vs 200 epochs on the black-white images also had worse results.

Selected Results:

Black-white images dataset, batch size of 32, epochs 100, lr 0.0005, 345 images

Black-white images dataset, batch size of 32, epochs 200, lr 0.0005, 345 images

Black-white images dataset, batch size of 32, epochs 1000, lr 0.0005, 345 images

Color images dataset, batch size of 32, epochs 500, lr 0.0005, 229 images

Color images dataset, batch size of 32, epochs 1000, lr 0.0005, 229 images



## Discussion

We created our own dataset using Google extensions Web Scraper and Tab Save to scrape tattoo images. We created two datasets - one for colored images (229 images) on various parts of human bodies, and one for black and white images (345 images) of clean tattoos on a white background. Some of the black-white images had some minimal red parts in addition to the black and white.

Regarding our networks, we followed an existing Pytorch tutorial to implement a specific type of GAN called DCGAN. After running this network on the provided tutorial dataset as well as our own datasets, we decided to change certain parameters and train and compare several new versions of the DCGAN model. 

We fed the raw images to the DCGAN we created and explored what tattoo images the DCGAN created. For the sake of time we decided to not further clean our images, though we did not receive very good results with our model. Instead, we tried tweaking the hyper-rparameter to see which changes provide the best results.

Our comparison focuses on the success of the different models on our two tattoo datasets. For this application of image generation, “success” must be determined by us humans by looking at the produced images and judging them based on what appears to be a clean image of something tattoo-like. Running the model on best sets did not provide results which we would call success. The colored images had much worse results compared to the black and white images, which can be explained by the huge difference of backgrounds of these images - color images were on different backgrounds, including different body parts, while the black-white images were on a clean white background.

We hope to be able to determine if the model is overfitting or not by judging the generated images for variety and uniqueness. We also hope that if we run the model several times, the generated images will look somewhat different each time. [Include actual results in a later iteration of this discussion.] Looking at existing tattoo generation models, we are currently unsure how our results compare. We will be able to have a clearer statement of comparison after running the model and producing images, which will be included in a later iteration of this discussion. [Add comments about why ours is better or worse or unidentifiable different in a later iteration of this discussion.]

On the colorful tattoo dataset with real human bodies, the network was unable to discern tattoos and backgrounds, failing to generate meaningful tattoos but images with random color lumps which seem to be  learned from human skins and backgrounds unrelated to tattoos. On the tattoo dataset with white backgrounds, the network was able to generate images that look like blurred tattoo images or real tattoo images with extremely low pixels. Although the network learned to generate images that roughly imitate the shapes and contours of a real tattoo image, it can’t generate distinct, meaningful lines but seemingly connected color patches with random pixel noise. 

The discriminator did a good job distinguishing generated tattoo images from real tattoo images, reaching almost 0 loss. The generator on the other hand, despite going down initially, tended to slightly go up as the training progressed and bump up and down. 

We suspect that the relatively small size of the tattoo dataset, compared with the original facial dataset, which contains 110342 images, mainly attributes to the drop down of performance. Methods to compensate for the limited amount of data help to increase the overall performance, e.g.increasing epochs to reach higher precisions of weights and biases, increasing learning rates to reduce overfitting, and reducing batch size to a more appropriate proportion of the whole dataset. 


## Ethical Discussion
Looking at the ethical issues that might be raised from this project there are a few things that we can think about. First, however, is looking at whether we should be doing this to begin with. On one hand, we believe we should be doing this because it is on a topic that is interesting to us for a Neural Networks class project. On the other hand, in the non-classroom setting of real life there should be deeper thought and more time devoted to acknowledging the artists whose original work our tattoo generator model uses for training. Art generation models are controversial because they can be seen as copying an artist’s work, stealing the very thing these artists devote their lives to.

After learning more about the ethics of art generation in general, we decided to pursue this project given that it likely wouldn't be a phenomenal tattoo generator and that it would not be widely used by the general public. That being said, there are 8 main ethical concerns that our group has discussed and wants to consider in this project and in its future iterations.

First, we consider the diversity of our team. We are not representative of all skin colors, we each come from cultures and families that don’t support tattoos, and none of us have tattoos. Thus, we all share similar ideals that may affect our considerations while working on this model.

Second, we consider the problem of artistic copyright. As we deal with artistic pieces, we must recognize that they are generated by tattoo artists who have copyright ownership over those artistic pieces. We need to make sure that the images in our dataset do not conflict with copyrights. It is also important to ask how Tattoo artists may feel with such a program, which is taking away part of their living and artistic value. 

Third, we consider the diversity of the tattooed bodies included in our dataset. Tattoos look different on different colors of skin, so our dataset needs to be diverse enough to include a relatively even percentage of general skin color categories. One thing we did in this project in an attempt to better train the model was create a dataset of only tattoo designs on a plain white background. So, we excluded skin color as a variable training on this dataset. Although this seemed to generate clearer tattoo designs, we would like to eventually get a diverse tattoo dataset large enough to generate designs of similar quality but on different colors of skin.

Fourth, we consider how our model might encourage young people to get tattoos. Having easy access to a computer program that freely generates tattoo images with a single click may encourage young people to get a tattoo at an earlier age than they might otherwise. However, this program may also reduce the cost of getting a tattoo because the design process is taken care of. So, it may increase the accessibility of getting a tattoo because it can reduce the financial burden, although minimally. 

Fifth, the ethical issue of indigenous culture comes into question. Our model must take into consideration that some cultures have a higher level meaning behind their tattoos rather than simply having them placed for self-pleasure. In certain cultures, a tattoo can symbolize maturity, while others can indicate a tribal connection, stage of life, age, genealogical connection, and other defining characteristics. Although this is a highly sensitive topic, our project aims to put a disclaimer that in the chance that our model does replicate or mimic art from an indigenous culture, we recommend users to research any potential related art they plan on using or distributing, and to carefully consider any offensive implications with for tattoos with potential tribal connections. 

Sixth, we need to consider how our program presents the risk of generating tattoos with offensive words/symbols. Our training dataset might include tattoos with offensive images, which we cannot thoroughly screen due to the limitations of our program’s scope. Thus, it is important to come up with measures to prevent the generation of offensive tattoos by our program, such as providing users with clear guidelines and warnings.

Seventh, a future iteration of this project would need to consider the words used to label tattoos. Currently, our model does not allow a user to describe the type of tattoo they want the model to generate. We do not include a text-based model, and we do not include labeled images. However, in a future iteration of this project we would love for this to be a capability. But new capabilities also bring new ethical concerns. We would need some way to ensure that people could not generate offensive tattoos, which stems from how tattoos are labeled in the training dataset. We would like to have some way to process the labeled data to remove anything that could create offensive content.

The eighth, and final ethical consideration (for now), involves language on tattoos in our training dataset. Since many existing tattoos may contain words of profanity, the data we train on may lead to tattoo generations that contain hateful and disrespectful words. We would like to ensure that generated tattoos avoid discriminating words or terms which are offensive to groups. 

Overall, we have many many ethical considerations at play within our tattoo generation project. While most were not feasible for our timeline and capabilities, we benefit greatly from thinking about them, writing about them, and proposing that they be used as this project is improved in the future.


## References
<a id="1">[1]</a>
Betin, Vasily. “Artificially Generated Tattoo.” Artificially Generated Tattoo, Medium, 12 Mar. 2020, https://medium.com/vasily-betin/artificially-generated-tattoo-2d5fbe0f5146. 

<a id="2">[2]</a>
Hood, Lonnie Lee. “Man Gets Tattoo of Art Created by Neural Network.” Man Gets Tattoo of Art Created By Neural Network, Futurism, 10 Apr. 2022, https://futurism.com/the-byte/tattoo-created-by-neural-network. 

<a id="3">[3]</a>
Schweitzer, Annette. “How Ai Can Help You Find the Perfect Tattoo.” Getting Good Ink: How AI Can Help You Find the Perfect Tattoo, NVIDIA Blog, 29 Sept. 2017, https://blogs.nvidia.com/blog/2017/09/29/find-the-perfect-tattoo/. 

<a id="4">[4]</a>
Vuksic, Goran. “Ai and Tattoos: How We Built a Neural Network for Tattoo Style Recognition.” AI and Tattoos: How We Built a Neural Network for Tattoo Style Recognition, Medium, 25 Aug. 2017, https://blog.tattoodo.io/ai-and-tattoos-how-we-built-a-neural-network-for-tattoo-style-recognition-6e641df99a05. 
