import numpy as np
import os.path as osp
from nltk.translate.bleu_score import corpus_bleu
from rdkit import RDLogger
from Levenshtein import distance as lev
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')
from fcd import get_fcd, load_ref_model, canonical_smiles
import warnings
warnings.filterwarnings('ignore')
def get_smis(filepath):
    print(filepath)
    with open(filepath) as f:
        lines = f.readlines()
    gt_smis= []
    op_smis = []
    for s in lines:
        if len(s)<3:
            continue
        s0,s1 = s.split(' || ')
        s0,s1 = s0.strip().replace('[EOS]','').replace('[SOS]','').replace('[X]','').replace('[XPara]','').replace('[XRing]',''),s1.strip()
        gt_smis.append(s1)
        op_smis.append(s0)
    return gt_smis,op_smis

def evaluate(gt_smis,op_smis):
    references = []
    hypotheses = []
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        gt_tokens = [c for c in gt]
        out_tokens = [c for c in out]
        references.append([gt_tokens])
        hypotheses.append(out_tokens)
    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    references = []
    hypotheses = []
    levs = []
    num_exact = 0
    bad_mols = 0
    for i, (gt, out) in enumerate(zip(gt_smis,op_smis)):
        hypotheses.append(out)
        references.append(gt)
        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
        except:
            bad_mols += 1
        levs.append(lev(out, gt))
    # Exact matching score
    exact_match_score = num_exact/(i+1)
    # Levenshtein score
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols/len(gt_smis)
    return bleu_score, exact_match_score, levenshtein_score, validity_score


def fevaluate(gt_smis,op_smis, morgan_r=2):
    outputs = []
    bad_mols = 0
    for n, (gt_smi,ot_smi) in enumerate(zip(gt_smis,op_smis)):
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)
            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append((gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    enum_list = outputs
    for i, (gt_m, ot_m) in enumerate(enum_list):
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def fcdevaluate(qgt_smis,qop_smis):
    gt_smis = []
    ot_smis = []
    for n, (gt_smi,ot_smi) in enumerate(zip(qgt_smis,qop_smis)):
        if len(ot_smi) == 0: ot_smi = '[]'
        gt_smis.append(gt_smi)
        ot_smis.append(ot_smi)
    model = load_ref_model()
    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]
    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    return fcd_sim_score

gt,op = get_smis('tempoutput.txt')
bleu_score, exact_match_score, levenshtein_score,_  = evaluate(gt,op)
validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fevaluate(gt,op)
fcd_metric_score = fcdevaluate(gt,op)
print(f'BLEU: {round(bleu_score, 3)}')
print(f'Exact: {round(exact_match_score, 3)}')
print(f'Levenshtein: {round(levenshtein_score, 3)}')
print(f'MACCS FTS: {round(maccs_sims_score, 3)}')
print(f'RDK FTS: {round(rdk_sims_score, 3)}')
print(f'Morgan FTS: {round(morgan_sims_score, 3)}')
print(f'FCD Metric: {round(fcd_metric_score, 3)}')
print(f'Validity: {round(validity_score, 3)}')