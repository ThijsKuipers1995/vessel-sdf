# vessel-sdf
Representing and generating semantically labeled vessels using self-supervised signed distance fields (SDFs).


### requirements

```
pytorch
open3d
trimesh
pymcubes
timm
```

### Usage

Follow the steps below to process the data and train models.

## Data preprocessing and setup

Download TopCoW 2024 and/or VascuSynth and place the datasets in the `datasets/data` directory.

Datasets can be processed into meshes and pointclouds by setting the `process_data` argument in their respective dataclasses (`datasets/topcow.py` and `datasets/vascusynth.py`) to `True` and initializing the dataset class (with default settings). Change the default paths in the dataclasses if you want to change the location the data is saved

## Training decoder and generator

Run `train_decoder.py` to start training. Use the `-h` flag to see all train settings. Same applies to `train_generator.py`. Here, a path to a decoder checkpoint must be specified. By default, checkpoints are saved as `checkpoints/{dataset_name}/[generator/decoder]/{model_name}/checkpoint-{epoch}.pth`.

## Generating shapes

Run `generate.py` to generate a shape. Generation settings, such as which models to use, can be specified in a separate config file. See `config_files/generate.yaml` for an example.