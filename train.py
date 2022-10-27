import argparse

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
from torch.optim.lr_scheduler import StepLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import json

import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return (
                torch.tensor(self.inputs[idx]["input_ids"]),
                torch.tensor(self.inputs[idx]["attention_mask"]),
                torch.tensor(self.inputs[idx]["token_type_ids"]),
            )
        else:
            return (
                torch.tensor(self.inputs[idx]["input_ids"]),
                torch.tensor(self.inputs[idx]["attention_mask"]),
                torch.tensor(self.inputs[idx]["token_type_ids"]),
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

        # preprocessing options
        self.del_special_symbol = True
        self.del_stopword = True
        self.del_dup_char = True

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, model_max_length=160
        )
        self.target_columns = ["label"]
        self.delete_columns = ["id"]

    def replaceSpecialSymbol(self, text):

        text = re.sub(pattern='…', repl='...', string=text)
        #text = re.sub(pattern='·', repl='.', string=text)
        text = re.sub(pattern='[’‘]', repl='\'', string=text)
        text = re.sub(pattern='[”“]', repl='\"', string=text)
        text = re.sub(pattern='‥', repl='..ㅤㅤ', string=text)
        text = re.sub(pattern='｀', repl='`', string=text)
        #print("이상한 특수기호 변형")
        #pattern = '[^\w\s]'
        #text = re.sub(pattern=pattern, repl='.', string=text)
        return text

    # 문자열에서 인접한 중복 문자를 제거하는 기능
    def removeDuplicates(self, text):
        chars = []
        prev = None
        for c in text:
            if prev != c:
                chars.append(c)
                prev = c
        return ''.join(chars)

    def cleanText(self, text):
        #pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' 
        #text = re.sub(pattern=pattern, repl='', string=text)
        #print("E-mail제거 : " , text , "\n")
        #pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        #text = re.sub(pattern=pattern, repl='', string=text)
        #print("URL 제거 : ", text , "\n")
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
        text = re.sub(pattern=pattern, repl='', string=text)
        #print("한글 자음 모음 제거 : ", text , "\n")
        #pattern = '<[^>]*>'        
        #text = re.sub(pattern=pattern, repl='', string=text)
        #print("태그 제거 : " , text , "\n")
        #pattern = r'\([^)]*\)'
        #text = re.sub(pattern=pattern, repl='', string=text)
        #print("괄호와 괄호안 글자 제거 :  " , text , "\n")
        #pattern = '[^\w\s]'   
        #text = re.sub(pattern=pattern, repl='', string=text)
        #print("특수기호 제거 : ", text , "\n" )
        text = text.strip()
        #print("양 끝 공백 제거 : ", text , "\n" )
        text = " ".join(text.split())
        #print("중간에 공백은 1개만 : ", text )
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
            self.train_dataset, batch_size=self.batch_size, shuffle=args.shuffle
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
    def __init__(self, model_name, lr, step_size=None, gamma=None):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()
        # self.loss_func = torch.nn.MSELoss()

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

        if self.step_size and self.gamma:
            scheduler = StepLR(optimizer, self.step_size, self.gamma)

            return [optimizer], [scheduler]

        return optimizer


def train(args, entity=None, project_name=None, wandb_check=True):

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
    )
    model = Model(args.model_name, args.learning_rate)

    if wandb_check:
        wandb.init(
            entity=entity,
            project=project_name,
            name=f"(batch:{args.batch_size},epoch:{args.max_epoch},lr:{args.learning_rate})",
        )
        wandb_logger = WandbLogger(project=project_name)
    else:
        wandb_logger = True

    # pytorch lightning model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson",
        dirpath=f"checkpoint/{project_name}/batch{args.batch_size}_epoch{args.max_epoch}_lr{args.learning_rate}",
        filename="{epoch:02d}-{val_pearson:.2f}",
        save_top_k=save_top_k,
        mode="min",
    )

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],  # checkpoint 설정 추가
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    # torch.save(model, "model.pt")


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="klue/roberta-base", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epoch", default=1, type=int)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--train_path", default="../data/train.csv")
    parser.add_argument("--dev_path", default="../data/dev.csv")
    parser.add_argument("--test_path", default="../data/dev.csv")
    parser.add_argument("--predict_path", default="../data/test.csv")
    parser.add_argument("--with_wandb", default=True, type=str2bool)
    parser.add_argument("--with_wandb_sweep", default=True, type=str2bool)
    args = parser.parse_args()

    project_name = args.model_name.split("/")[-1]
    entity = "naver-nlp-07"
    save_top_k = 3

    with open("config.json", "r") as f:
        config = json.load(f)

    if args.with_wandb:
        wandb.login(key=config["key"])  ##insert key

        if args.with_wandb_sweep == True:
            # Sweep 할 대상
            sweep_config = config["sweep_config"]

            def sweep_train(config=None):
                wandb.init(
                    config=config,
                )
                config = wandb.config
                # dataloader와 model을 생성합니다.
                dataloader = Dataloader(
                    args.model_name,
                    config.batch_size,
                    args.shuffle,
                    args.train_path,
                    args.dev_path,
                    args.test_path,
                    args.predict_path,
                )
                model = Model(
                    args.model_name,
                    config.learning_rate,
                    config.step_size,
                    config.gamma,
                )
                # wandb logger for pytorch lightning Trainer
                wandb_logger = WandbLogger(project=project_name)

                # pytorch lightning model checkpoint
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_pearson",
                    dirpath=f"checkpoint/{project_name}/batch{config.batch_size}_epoch{args.max_epoch}_lr{config.learning_rate}",
                    filename="{epoch:02d}-{val_pearson:.2f}",
                    save_top_k=save_top_k,
                    mode="min",
                )

                # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
                trainer = pl.Trainer(
                    gpus=1,
                    max_epochs=args.max_epoch,
                    log_every_n_steps=1,
                    logger=wandb_logger,
                    callbacks=[checkpoint_callback],  # checkpoint 설정 추가
                )

                # Train part
                trainer.fit(model=model, datamodule=dataloader)
                trainer.test(model=model, datamodule=dataloader)

            # 학습이 완료된 모델을 저장합니다. 어차피 checkpoint로 마지막 3개를 저장하니까 마지막은 중복저장됨.
            # sweep.agent를 사용해서 학습 시작
            sweep_id = wandb.sweep(
                sweep=sweep_config, entity=entity, project=project_name
            )
            wandb.agent(
                sweep_id=sweep_id, function=sweep_train, count=2
            )  # Sweep을 몇번 실행할 지 선택
            # -------------------------------------------------------------------------------------

        else:
            train(
                args=args,
                entity=entity,
                project_name=project_name,
                wandb_check=args.with_wandb,
            )

    else:
        # dataloader와 model을 생성합니다.
        train(args=args, project_name=project_name, wandb_check=args.with_wandb)
