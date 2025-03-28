import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import matplotlib.pyplot as plt
from model.aa_model import get_model

import numpy as np
import pandas as pd
from dataset import encode_sequence
def main():
    # 설정 값 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 초기화 및 가중치 로드
    model = get_model().to(device)
    model.load_state_dict(torch.load("checkpoints/model_weights_epoch_501.pth", map_location=device))
    model.eval()
    
    # 누락된 데이터 로드
    missing_df = pd.read_csv("missing.csv")
    max_len = 10  # `dataset.py`에서 사용된 최대 길이
    missing_features = np.array([encode_sequence(seq, max_len) for seq in missing_df["AA"]])

    # 누락된 데이터 예측
    missing_predictions = []
    with torch.no_grad():
        for i in range(0, len(missing_features), 1024):  # 배치 크기 1024로 처리
            batch = torch.tensor(missing_features[i:i+1024], dtype=torch.long).to(device)
            preds = model(batch).cpu().numpy().flatten()
            missing_predictions.extend(preds)

    # 히스토그램 생성
    plt.figure(figsize=(8, 6))
    plt.hist(missing_predictions, bins=50, color="blue", alpha=0.7, edgecolor="black")
    plt.title("Histogram of Predicted Missing Production Values")
    plt.xlabel("Predicted Production Value")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig("missing_predictions_histogram.png", dpi=300)
    plt.close()

    print("Missing data predictions and histogram saved.")

if __name__ == '__main__':
    main()