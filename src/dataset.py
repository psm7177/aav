import pandas as pd
import torch

from torch.utils.data import Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}

# AA 서열을 정수 인덱스 리스트로 변환
def encode_sequence(seq, max_len):
    encoded = [AA_TO_IDX.get(aa, 0) for aa in seq]
    encoded += [0] * (max_len - len(encoded))
    return encoded[:max_len]

class AAVDataset(Dataset):
    def __init__(self, csv_file, aa_column="AA", target_column="Production", max_len=10):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len

        # MinMax 정규화
        # self.scaler = MinMaxScaler()
        # self.data[target_column] = self.scaler.fit_transform(self.data[[target_column]])

        self.X = torch.tensor([encode_sequence(seq, max_len) for seq in self.data[aa_column]], dtype=torch.long)
        self.y = torch.tensor(self.data[target_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataset():
    csv_file = "production.csv"
    dataset = AAVDataset(csv_file)
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    return train_dataset, test_dataset