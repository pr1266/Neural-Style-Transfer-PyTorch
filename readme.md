# Build your own Artistic Image!
one of the most beautifull applications of deep neural networks is neural style transfer. let's take an example:
suppose that we have an source image that we want to paint it in the style of a famous painting. so we can init the target image with a random tensor, and then change it through train stage and make it similar to combination of both source and style image. whatever we consider a larger coefficient for style-content loss, output will be more similar to style image and more different from source content image. the idea of this work is to feed both source and style image to a pre-trained network (like VGG19) and capture certain convolution layers output, calculate loss according to both style and source image using Gram matrix.

# Result
### Built With
* [![PyTorch][torchlogo]][torchurl]


[torchlogo]: https://img.shields.io/badge/pytorch-ff8200?style=for-the-badge&logo=PyTorch&logoColor=white
[torchurl]: https://pytorch.org/