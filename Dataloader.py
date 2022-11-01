import torch
from torch.optim.lr_scheduler import StepLR

import transformers
import pandas as pd
import pytorch_lightning as pl

from tqdm.auto import tqdm

import re
from sklearn.model_selection import KFold

from Dataset import Dataset

class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        k: int = 0,
        split_seed: int = 12345,
        num_splits: int = 3,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # preprocessing options
        self.del_special_symbol = True
        self.del_stopword = False
        self.del_dup_char = False

        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, model_max_length=256
        )
        self.target_columns = ["label"]
        self.delete_columns = ["id"]

    def replaceSpecialSymbol(self, text):

        text = re.sub(pattern="…", repl="...", string=text)
        # text = re.sub(pattern='·', repl='.', string=text)
        text = re.sub(pattern="[’‘]", repl="'", string=text)
        text = re.sub(pattern="[”“]", repl='"', string=text)
        text = re.sub(pattern="‥", repl="..ㅤㅤ", string=text)
        text = re.sub(pattern="｀", repl="`", string=text)
        # print("이상한 특수기호 변형")
        # pattern = '[^\w\s]'
        # text = re.sub(pattern=pattern, repl='.', string=text)
        # print("특수 기호 통일")
        return text

    # 문자열에서 인접한 중복 문자를 제거하는 기능
    def removeDuplicates(self, text):
        chars = []
        prev = None
        for c in text:
            if prev != c:
                chars.append(c)
                prev = c
        return "".join(chars)

    def deleteStopword(self, text):
        # pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("E-mail제거 : " , text , "\n")
        # pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("URL 제거 : ", text , "\n")
        pattern = "([ㄱ-ㅎㅏ-ㅣ]+)"
        text = re.sub(pattern=pattern, repl="", string=text)
        text = text.strip()
        # print("양 끝 공백 제거 : ", text , "\n" )
        text = " ".join(text.split())
        # print("중간에 공백은 1개만 : ", text )
        return text

    def cleanText(self, text):
        # 특수 문자 처리
        if self.del_special_symbol is True:
            text = self.replaceSpecialSymbol(text)

        # 불용어 처리를 진행합니다.
        if self.del_stopword is True:
            text = self.replaceSpecialSymbol(text)

        # 반복되는 문자 제거
        if self.del_dup_char is True:
            text = self.replaceSpecialSymbol(text)

        return text
      

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            # 텍스트 정규화
            clean_sentence_1 = self.cleanText(item["sentence_1"])
            clean_sentence_2 = self.cleanText(item["sentence_2"])

            # 토큰화
            outputs = self.tokenizer(
                clean_sentence_1,
                clean_sentence_2,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
            data.append(outputs)
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 데이터 준비
            total_data = pd.read_csv(self.train_path)
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size
        )

