"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from mytokenizers import SimpleSmilesTokenizer
import argparse
import os, json
from rdkit import Chem
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from mydatasets import get_dataloader,ChEBIdataset

def main():
    set_seed(121)
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    from mytokenizers import regexTokenizer
    tokenizer = regexTokenizer()
    model = TransformerNetModel2(
        in_channels=32,  # 3, DEBUG**
        # deep_channels = 10,
        model_channels=128,
        dropout=0.1,
        use_checkpoint=False,
        config_name='bert-base-uncased',
        training_mode='e2e',
        vocab_size=len(tokenizer),
        experiment_mode='lm',
        logits_mode=1,
        hidden_size = 1024,
        num_attention_heads=16,
        num_hidden_layers = 12,
    )
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0,2000,10)],
        betas=gd.get_named_beta_schedule('sqrt', 2000),
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',
        training_mode='e2e',
    )

    print(args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    model.to(dist_util.dev())
    model.eval() # DEBUG

    logger.log("sampling...")
    print(args.num_samples)
    # model3 = get_weights(model2, args)
    print('--'*30)
    print('loading {} set'.format(args.split))
    print('--'*30)

    train_dataset = ChEBIdataset(
        dir='../../datasets/SMILES/',
        smi_tokenizer=tokenizer,
        split=args.split,
        replace_desc=False
        # pre = pre
    )
    print('DATASETINFO-----------------------------')
    print(len(train_dataset),(train_dataset[0]['desc_state'].shape))
    desc = [(train_dataset[i]['desc_state'],train_dataset[i]['desc_mask'],train_dataset[i]['smiles']) for i in range(args.num_samples)]
    answer = [i[2] for i in desc]
    model3 = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model3.requires_grad = False


    
    allsample = []
    num_done = 0
    while num_done < args.num_samples:
        idend = min(num_done+args.batch_size,args.num_samples)
        print('acquiring  {} : {}'.format(num_done,idend))
        desc_state = th.concat([i[0] for i in desc[num_done:idend]],dim=0)
        desc_mask = th.concat([i[1] for i in desc[num_done:idend]],dim=0)
        
        model_kwargs = {}
        print('use_ddim:{}',args.use_ddim)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_shape = (idend-num_done, 256, model.in_channels)
        print(sample_shape)
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn = None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
            progress = True,
            desc = (desc_state,desc_mask)
        )
        allsample.append(sample)
        num_done = idend
    sample = th.concat(allsample,dim=0)
    print('decoding for e2e', )
    print(sample.shape)
    x_t = th.tensor(sample).cuda()
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    sample = sample.squeeze(-1)
    print(sample)
    from mytokenizers import regexTokenizer
    tokenizer = regexTokenizer()
    c = tokenizer.decode(sample)
    with open(args.outputdir,'w') as f:
        for i,x in enumerate(c):
            if i==0:
                print(x)
            f.write(x.replace('[PAD]','')+'   ||   '+answer[i]+'\n')


    with open(args.outputdir) as f:
        allsmiles = [k.strip().split('||')[0].strip().replace('[EOS]','').replace('[SOS]','') for k in f.readlines()]
    f = open('../../tempbadmols.txt','w')
    for cnt,s in enumerate(allsmiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            f.write(str(cnt)+'\t'+s+'\n')
    f.close()

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, clamp='clamp',split = 'test',
                         model_path='../../checkpoints/PLAIN_ema_0.9999_200000.pt',
                         use_ddim=False,
                         batch_size =64,num_samples=3300,top_p =1.0,out_dir='generation_outputs',
                         outputdir='../../textguidtry_256_final.txt'
                         )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
