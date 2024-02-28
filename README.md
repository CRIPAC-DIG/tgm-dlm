# Text-Guided Molecule Generation with Diffusion Language Model

![tgmdlm](pics/tgmdlm.png)

Accepted by AAAI-24, check our paper at https://arxiv.org/abs/2402.13040.
If you find it useful, please consider citing:
```
@article{gong2024text,
  title={Text-Guided Molecule Generation with Diffusion Language Model},
  author={Haisong Gong and Qiang Liu and Shu Wu and Liang Wang},
  journal={arXiv preprint arXiv:2402.13040},
  year={2024}
}
```

---

（This code is based on https://github.com/XiangLi1999/Diffusion-LM and https://github.com/blender-nlp/MolT5）

---

## Preparation

1. Install Package `cd TGMDLMCODE; pip install -e improved-diffusion/; pip install -e transformers/`.
2. Download [Scibert](https://huggingface.co/allenai/scibert_scivocab_uncased) and put it into file `scibert`.


## Training
1. `cd improved-diffusion; cd scripts`
2. Encode text input `python process_text.py -i train_val_256; python process_text.py -i test`
3. Train model for Phase One: `python train.py`
4. Train model for Phase Two: `python train_correct_withmask.py`

## Sampling
1. `python text_sample.py; python post_sample.py` The final file `OURMODEL_OUTPUT.txt` is our output.

## Evaluation
you can evaluate all metrics except for Text2Mol by runnning `ev.py`. For Text2Mol please go to [MolT5](https://github.com/blender-nlp/MolT5) for more details.