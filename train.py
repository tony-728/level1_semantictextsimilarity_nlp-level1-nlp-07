# import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import json

from Dataloader import Dataloader
from Model import Model




def train(config, entity=None, project_name=None, wandb_check=True):

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        config["model_name"],
        config["batch_size"],
        config["shuffle"],
        config["train_path"],
        config["dev_path"],
        config["test_path"],
        config["predict_path"],
    )
    model = Model(
        model_name=config["model_name"],
        batch_size = config["batch_size"], 
        max_epoch = config["max_epoch"], 
        learning_rate = config["learning_rate"],
        step_size=config["step_size"],
        gamma=config["gamma"],
    )

    if wandb_check:
        wandb.init(
            entity=entity,
            project=project_name,
            name=f"(batch:{config['batch_size']},epoch:{config['max_epoch']},lr:{config['learning_rate']})",
        )
        wandb_logger = WandbLogger(project=project_name)
    else:
        wandb_logger = True

    # pytorch lightning model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson",
        dirpath=f"checkpoint/{project_name}/batch{config['batch_size']}_epoch{config['max_epoch']}_lr{config['learning_rate']}",
        filename="{epoch:02d}-{val_pearson:.2f}",
        save_top_k=save_top_k,
        mode="max",
    )

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config["max_epoch"],
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
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    with open("config.json", "r") as f:
        train_config = json.load(f)

    project_name = train_config["model_name"].split("/")[-1]
    entity = "naver-nlp-07"
    save_top_k = 5

    if train_config["with_wandb"]:
        wandb.login(key=train_config["key"])  ##insert key

        if train_config["with_wandb_sweep"] == True:
            # Sweep 할 대상
            sweep_config = train_config["sweep_config"]

            def sweep_train(config=None):
                wandb.init(
                    config=config,
                )
                config = wandb.config
                # dataloader와 model을 생성합니다.
                dataloader = Dataloader(
                    train_config["model_name"],
                    config.batch_size,
                    train_config["shuffle"],
                    train_config["train_path"],
                    train_config["dev_path"],
                    train_config["test_path"],
                    train_config["predict_path"],
                )
                model = Model(
                    train_config["model_name"],
                    config.batch_size,
                    train_config["max_epoch"],
                    config.learning_rate,
                    config.step_size,
                    config.gamma,
                )
                # wandb logger for pytorch lightning Trainer
                wandb_logger = WandbLogger(project=project_name)

                # pytorch lightning model checkpoint
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_pearson",
                    dirpath=f"checkpoint/{project_name}/batch{config.batch_size}_epoch{train_config['max_epoch']}_lr{config.learning_rate}",
                    filename="{epoch:02d}-{val_pearson:.2f}",
                    save_top_k=save_top_k,
                    mode="max",
                )

                # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
                trainer = pl.Trainer(
                    gpus=1,
                    max_epochs=train_config["max_epoch"],
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
                config=train_config,
                entity=entity,
                project_name=project_name,
                wandb_check=train_config["with_wandb"],
            )

    else:
        # dataloader와 model을 생성합니다.
        train(
            config=train_config,
            project_name=project_name,
            wandb_check=train_config["with_wandb"],
        )
