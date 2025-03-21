import pandas as pd
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from sklearn.metrics import r2_score

from dataset import AAVDataset
from model.aa_model import get_model


def main():
    # 데이터셋 & DataLoader 설정
    csv_file = "production.csv"
    dataset = AAVDataset(csv_file)
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 초기 학습률 0.0005
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    # 학습 루프
    num_epochs = 501
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        
        for i, (X_batch, y_batch) in enumerate(pbar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f'Loss: {total_loss / (i + 1)}')

        train_losses.append(total_loss / len(train_loader))
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
        
        # ---- 테스트 데이터로 예측 ----
        model.eval()
        real_values, pred_values = [], []
        test_loss = 0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device).unsqueeze(1)
                y_pred = model(X_test)
                loss = criterion(y_pred, y_test)
                test_loss += loss.item()
                
                real_values.extend(y_test.cpu().numpy().flatten())
                pred_values.extend(y_pred.cpu().numpy().flatten())
        
        test_losses.append(test_loss / len(test_loader))
        
        # R^2 값 계산
        r2 = r2_score(real_values, pred_values)
        print(f"Epoch {epoch+1} - R^2 Score: {r2:.4f}")
        # ---- 그래프 저장 ----
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(real_values, pred_values, alpha=0.5, color="blue")
        ax.plot([-10, 10], [-10, 10], linestyle="--", color="red")
        ax.set_xlabel("Real Production Value")
        ax.set_ylabel("Predicted Production Value")
        ax.set_title(f"Epoch {epoch+1} - Real vs Predicted")
        ax.grid()
        fig.savefig(f"checkpoints/epoch_{epoch+1}_plot.png")
        plt.close(fig)

        # 학습 및 테스트 손실 그래프 저장
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train & Test Loss Over Epochs")
        plt.legend()
        plt.savefig("train_test_loss.png")
        plt.close()

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_weights_epoch_{epoch+1}.pth")


    # 모델 가중치 저장
    torch.save(model.state_dict(), "model_weights.pth")
    print("Training Complete! Model weights saved as 'model_weights.pth'")

if __name__ == "__main__":
    main()