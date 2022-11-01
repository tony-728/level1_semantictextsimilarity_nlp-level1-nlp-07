import pandas as pd
import matplotlib.pyplot as plt

import transformers
import torch
from torch.optim.lr_scheduler import StepLR
import torchmetrics
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(
        self, 
        model_name,
        batch_size,
        max_epoch,
        learning_rate,
        step_size=None, 
        gamma=None):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name

        self.epoch_num = 0

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate

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
        val_pearson = torchmetrics.functional.pearson_corrcoef(
            logits.squeeze(), y.squeeze()
        )

        self.log(
            "val_pearson",
            val_pearson,
        )

        return {"logits": logits, "val_pearson": val_pearson}

    def validation_epoch_end(self, outputs):
        val_pearson = [x["val_pearson"] for x in outputs]
        dev_pred = [x["logits"].squeeze() for x in outputs]
        # print("처리 전 : ", dev_pred)
        dev_pred = list(round(float(i), 1) for i in torch.cat(dev_pred))
        # print("처리 후 : ", dev_pred)

        if len(dev_pred) == 550:  # dev 데이터 개수 확인
            output = pd.read_csv("../data/dev.csv")
            output["pred"] = dev_pred

            X = output.label.values
            Y = output.pred.values

            plt.scatter(X, Y, alpha=0.5)
            plt.title(
                f"Label/Pred, pearson mean : {round(float(sum(val_pearson)/len(val_pearson)), 5)}"
            )
            plt.xlabel("Label")
            plt.ylabel("Pred")

            plt.savefig(
                f"batch:{self.batch_size},epoch:{self.max_epoch},lr:{self.learning_rate}, epoch:{self.epoch_num}.png",
                dpi=200,
            )
            plt.clf()  # 그래프를 겹쳐서 보고 싶으면 주석처리함
            self.epoch_num += 1

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.step_size and self.gamma:
            scheduler = StepLR(optimizer, self.step_size, self.gamma)

            return [optimizer], [scheduler]

        return optimizer
