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

The example textures go in the folder **Textures**. The output file **params.pytorch** contains the trained parameters of the generator network.

## Sample

### Texture Synthesis

```
python3 main.py  --pretrained 'Trained_models/<folder_containing_model>' --sample_size 512 --n_samples 3
```
The python script loads the trained parameters and synthesizes **n_samples** of a squared texture of size **sample_size**.


