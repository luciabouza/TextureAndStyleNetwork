# TextureNets_implementation

PyTorch implementation of the texture synthesis model in [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](https://arxiv.org/abs/1603.03417) of Ulyanov et al.

Based on 
- Jorge Gutierrez's [code](https://github.com/JorgeGtz/TextureNets_implementation)
- Notbook texture synthesis [colab](https://colab.research.google.com/github/luciabouzaheguerte/NotebooksProject/blob/master/ImageGeneration/CNN_Texture_Synthesis_with_solution.ipynb)


## Training

### Texture Synthesis

```
python3 main.py --train_texture true --input_texture tannat.jpg --max_iter 5000 --batch_size 8 --learning_rate 0.001 --lr_adjust 750 --lr_decay_coef 0.66
```

### Style Transfer

```
python main.py --train_style true --input_content Etretat.jpg --input_style van_gogh.jpg --max_iter 20000 --batch_size 3 --learning_rate 0.001 --lr_adjust 2000 --lr_decay_coef 0.8 --train_data /mnt/data/shared_datasets/imagenet/imagenet/imagenet/train
```

The example textures go in the folder **Textures**. 
The example styles go in the folder **Styles**. 
The example contents go in the folder **Contents**. 

The output file ***params.pytorch** contains the trained parameters of the generator network.

## Sample

### Texture Synthesis

```
python3 main.py  --pretrained 'Trained_models/<folder_containing_model>' --sample_size 512 --n_samples 3
```

The python script loads the trained parameters and synthesizes **n_samples** of a squared texture of size **sample_size**.

### Style Transfer

```
python3 main.py  --pretrained_style 'Trained_models/<folder_containing_model>' --input_content Etretat.jpg --sample_size 512 
```

The python script loads the trained parameters and creates a image with the content of **input_content** and the style learned by the network. The result will be a squared image of **sample_size**. The content image must go in the folder *Contents*. 


