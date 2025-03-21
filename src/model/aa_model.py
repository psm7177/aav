import torch
import torch.nn as nn
import math

from dataset import AMINO_ACIDS

class AAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, max_len, num_transformer_layers=3, num_heads=4):
        super(AAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.max_len = max_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Sigmoid()  # [0,1] 범위로 정규화
        )

        # Positional Encoding 초기화
        self.positional_encoding = self._get_positional_encoding(embed_dim, max_len + 1)

    def _get_positional_encoding(self, embed_dim, max_len):
        """사인-코사인 방식의 Positional Encoding 생성"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 사인 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 코사인 함수 적용
        
        pe = pe.unsqueeze(0)  # 배치 차원 추가
        return pe

    def forward(self, x):
        # 'cls' 토큰을 추가하기 위한 작업
        batch_size = x.size(0)
        cls_token = torch.zeros(batch_size, 1).long().to(x.device)  # [batch_size, 1] 크기의 cls 토큰 (0은 pad_idx로 가정)
        x = torch.cat([cls_token, x], dim=1)  # 입력 텍스트에 cls 토큰 추가

        # 임베딩 처리
        x = self.embedding(x)

        # Positional Encoding 추가
        positional_encoding = self.positional_encoding[:, :x.size(1)].to(x.device)

        # x와 positional encoding의 크기를 맞추기 위해 추가
        x = x + positional_encoding  # Positional Encoding 추가

        # Transformer 처리
        x = self.transformer(x)

        # 'cls' 토큰의 출력만 가져오기 (첫 번째 위치)
        cls_token_output = x[:, 0, :]

        # 최종 예측
        x = self.fc(cls_token_output)
        return x
    
def get_model():
    vocab_size = len(AMINO_ACIDS) + 1
    embed_dim = 16
    hidden_dim = 64
    output_dim = 1
    max_len = 10
    num_transformer_layers = 3
    num_heads = 4
    
    return AAModel(vocab_size, embed_dim, hidden_dim, output_dim, max_len, num_transformer_layers, num_heads)