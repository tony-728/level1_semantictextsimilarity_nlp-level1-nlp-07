
import json

import pandas as pd

import torch
import pytorch_lightning as pl

from Model import Model
from Dataloader import Dataloader


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

    model = Model(
        model_name = train_config["model_name"], 
        batch_size = train_config["batch_size"], 
        max_epoch = train_config["max_epoch"], 
        learning_rate = train_config["learning_rate"],
    )

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
