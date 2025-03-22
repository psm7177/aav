
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import get_dataset
from model.aa_model import get_model

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

def main():
    # 설정 값 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델 초기화 및 가중치 로드
    model = get_model().to(device)
    model.load_state_dict(torch.load("checkpoints/model_weights_epoch_501.pth", map_location=device))
    model.eval()

    _, test_dataset = get_dataset()
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 테스트 및 예측 수행
    real_values, pred_values = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device).unsqueeze(1)
            y_pred = model(X_test)
            
            real_values.extend(y_test.cpu().numpy().flatten())
            pred_values.extend(y_pred.cpu().numpy().flatten())

    real_values = np.array(real_values)
    pred_values = np.array(pred_values)

    sample_indices = np.random.choice(len(real_values), size=10000, replace=False)
    x_sample = real_values[sample_indices]
    y_sample = pred_values[sample_indices]

    # 밀도 계산
    kernel = gaussian_kde(np.vstack([x_sample, y_sample]))
    density = kernel(np.vstack([real_values, pred_values]))  # 전체 데이터 밀도 계산
    # 결과 시각화
    r2 = r2_score(real_values, pred_values)
    # 그래프 설정
    plt.figure(figsize=(6, 6))

    # 산점도 (밀도 기반 색상 적용)
    plt.scatter(real_values, pred_values, c=density, cmap="inferno", edgecolor="none", s=3, rasterized=True)
    # ax.scatter(x_both, y_both, c=c_both, cmap=mpl.cm.inferno, s=0.2, edgecolor='none', rasterized=True)
    # 이상적인 y = x 선 (데이터 범위 기반)
    min_val, max_val = min(real_values.min(), pred_values.min()), max(real_values.max(), pred_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")

    # 축 및 제목 설정
    plt.xlabel("Real Production Value")
    plt.ylabel("Predicted Production Value")
    plt.title(f"Test Results - Real vs Predicted R^2: {r2:.4f}")
    plt.grid()

    # 컬러바 추가 (밀도 정보)
    cbar = plt.colorbar()
    cbar.set_label("Density")

    # 고해상도 저장 및 출력
    plt.savefig('vailidation.png', dpi=300)

if __name__ == '__main__':
    main()