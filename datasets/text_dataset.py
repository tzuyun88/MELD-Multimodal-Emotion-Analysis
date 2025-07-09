from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class TextEmotionDataset(Dataset):
    def __init__(self, csv_path, label_map, max_len=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        utterance = self.df.iloc[idx]['Utterance']
        emotion = self.df.iloc[idx]['Emotion']

        # 編碼輸入文字
        encoding = self.tokenizer(
            utterance,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = self.label_map[emotion]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

