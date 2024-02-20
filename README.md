This code is based on https://github.com/XiangLi1999/Diffusion-LM and https://github.com/blender-nlp/MolT5

# Preparation

1. Install Package `cd TGMDLMCODE; pip install -e improved-diffusion/; pip install -e transformers/`.
2. Download [Scibert](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it into file `scibert`.


# Training
1. `cd improved-diffusion; cd scripts`
2. Encode text input `python process_text.py -i train_val_256; python process_text.py -i test`
3. Train model for Phase One: `python train.py`
4. Train model for Phase Two: `python train_correct_withmask.py`

# Sampling
1. `python text_sample.py; python post_sample.py` The final file `OURMODEL_OUTPUT.txt` is our output.

# Evaluation
you can evaluate all metrics except for Text2Mol by runnning `ev.py`. For Text2Mol please go to [MolT5](https://github.com/blender-nlp/MolT5) for more details.