import os

from dataset import decode_sequence

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
from model.aa_model import get_model

import matplotlib.pyplot as plt

import csv

from Bio.Data import CodonTable
import random

back_table = {}
for codon, aa in CodonTable.unambiguous_dna_by_name["Standard"].forward_table.items():
    if aa not in back_table:
        back_table[aa] = []
    back_table[aa].append(codon)

def amino_acids_to_dna_biopython(amino_acid_sequence: str) -> str:
    """
    아미노산 서열을 DNA 서열로 변환합니다 (역번역).
    Biopython의 표준 코돈 표를 사용합니다.
    """
    # 표준 코돈 테이블 불러오기

    dna_sequence = []
    for amino_acid in amino_acid_sequence:
        codons = back_table.get(amino_acid)
        if not codons:
            raise ValueError(f"Invalid amino acid: {amino_acid}")
        
        # 가능한 코돈 중 무작위로 선택
        dna_sequence.append(random.choice(codons))
    
    return ''.join(dna_sequence)

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 초기화 및 가중치 로드
    model = get_model().to(device)
    model.load_state_dict(torch.load("checkpoints/model_weights_epoch_501.pth", map_location=device))
    model.eval()

    x_seqs = []
    y_preds = []
    with torch.no_grad():
        for i in range(100):
            x = torch.zeros(1024, 10, dtype=torch.int).to(device)
            x[:, :7] = torch.randint(1, 21, (1024 ,7)).to(device)
            
            y_pred = model(x)

            x_seqs.extend(x.cpu().numpy())
            y_preds.extend(y_pred.cpu().numpy().flatten())

    x_seqs  = np.array([decode_sequence(x) for x in x_seqs])
    y_preds = np.array(y_preds)


    top = y_preds.argsort()[::-1][:10000]

    with open("generation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AA", "DNA", "Production"])
        for x, y in zip(x_seqs[top], y_preds[top]):
            dna = amino_acids_to_dna_biopython(x)
            writer.writerow([x, dna, y])

    plt.hist(y_preds, bins=50)
    plt.title("Histogram of Predicted Production Values")
    plt.xlabel("Predicted Production Value")
    plt.ylabel("Frequency")
    plt.savefig("generation.png")

if __name__ == '__main__':
    generate()