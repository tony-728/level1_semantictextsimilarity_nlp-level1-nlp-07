import torch

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

