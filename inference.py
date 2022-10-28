import json

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import re


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        input_ids = self.inputs[idx].get("input_ids")
        attention_mask = self.inputs[idx].get("attention_mask")
        token_type_ids = self.inputs[idx].get("token_type_ids")

        if len(self.targets) == 0:
            if token_type_ids == None:
                return (
                    torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                )
            else:
                return (
                    torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(token_type_ids),
                )

        else:
            if token_type_ids == None:
                return (
                    torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                ), torch.tensor(self.targets[idx])
            else:
                return (
                    torch.tensor(input_ids),
                    torch.tensor(attention_mask),
                    torch.tensor(token_type_ids),
                ), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


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

        self.del_special_symbol = True
        self.del_stopword = True
        self.del_dup_char = True

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, model_max_length=160
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

    def cleanText(self, text):
        # pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("E-mail제거 : " , text , "\n")
        # pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("URL 제거 : ", text , "\n")
        pattern = "([ㄱ-ㅎㅏ-ㅣ]+)"
        text = re.sub(pattern=pattern, repl="", string=text)
        # print("한글 자음 모음 제거 : ", text , "\n")
        # pattern = '<[^>]*>'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("태그 제거 : " , text , "\n")
        # pattern = r'\([^)]*\)'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("괄호와 괄호안 글자 제거 :  " , text , "\n")
        # pattern = '[^\w\s]'
        # text = re.sub(pattern=pattern, repl='', string=text)
        # print("특수기호 제거 : ", text , "\n" )
        text = text.strip()
        # print("양 끝 공백 제거 : ", text , "\n" )
        text = " ".join(text.split())
        # print("중간에 공백은 1개만 : ", text )
        return text

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            # 특수 문자 처리
            if self.del_special_symbol is True:
                item["sentence_1"] = self.replaceSpecialSymbol(item["sentence_1"])
                item["sentence_2"] = self.replaceSpecialSymbol(item["sentence_2"])

            # 불용어 처리를 진행합니다.
            if self.del_stopword is True:
                item["sentence_1"] = self.cleanText(item["sentence_1"])
                item["sentence_2"] = self.cleanText(item["sentence_2"])

            # 반복되는 문자 제거
            if self.del_dup_char is True:
                item["sentence_1"] = self.removeDuplicates(item["sentence_1"])
                item["sentence_2"] = self.removeDuplicates(item["sentence_2"])

            outputs = self.tokenizer(
                item["sentence_1"],
                item["sentence_2"],
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
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
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


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(*x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    with open("config.json", "r") as f:
        train_config = json.load(f)

    project_name = train_config["model_name"].split("/")[-1]

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        train_config["model_name"],
        train_config["batch_size"],
        train_config["shuffle"],
        train_config["train_path"],
        train_config["dev_path"],
        train_config["test_path"],
        train_config["predict_path"],
    )

    model = Model(train_config["model_name"], train_config["learning_rate"])

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    checkpoint_path = f"checkpoint/{project_name}/batch64_epoch5_lr1e-05/epoch=03-val_pearson=0.92.ckpt"
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=train_config["max_epoch"],
        log_every_n_steps=1,
    )

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    # pytorch lightning checkpoint load
    # model = torch.load("model.pt")
    model = Model.load_from_checkpoint(checkpoint_path)
    # -------------------------------------------------------------------------------------

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)
