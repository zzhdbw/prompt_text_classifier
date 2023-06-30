#软模板prompt使用bert编码模板
import torch
from transformers import BertModel
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM, BertForPreTraining, BertForQuestionAnswering, \
    BertForSequenceClassification
import os
import copy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_data(file, num=None):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    for data in all_data:
        data_s = data.split("\t")
        if len(data_s) != 2:
            continue

        text, label = data_s
        all_text.append(text)
        all_label.append(int(label))

    if num is None:
        return all_text, all_label
    else:
        return all_text[:num], all_label[:num]


class PDataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, index_2_label, prompt_num, mask_position, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.index_2_label = index_2_label

        self.prompt_len = prompt_num
        self.mask_position = mask_position
        self.prompt_content = self.generate_template()
        self.max_len = max_len

    def generate_template(self):
        prompt_template = []

        p_n = 0
        while p_n < self.prompt_len:
            if p_n in self.mask_position:
                prompt_template.append("[MASK]")

            prompt_template.append(f"[prompt_{p_n}]")
            p_n += 1
        return "".join(prompt_template)

    def __getitem__(self, index):
        text = self.all_text[index]

        label = self.all_label[index]
        label_text = self.index_2_label[label]

        text_prompt = self.prompt_content + text
        return text_prompt, label, label_text

    # ('[MASK][prompt_0][MASK][prompt_1][prompt_2][prompt_3][prompt_4][prompt_5][prompt_6][prompt_7][prompt_8][prompt_9]中华女子学院：本科层次仅1专业招男生',3, '教育', 30)
    def pro_batch_data(self, batch_data):
        batch_text, batch_label, batch_label_text = zip(*batch_data)

        batch_text_idx = []
        batch_label_idx = []
        batch_text_len = []

        # 句长统计
        for text_prompt in batch_text:
            text_idx = self.tokenizer.encode(text_prompt, add_special_tokens=True)
            batch_text_len.append(len(text_idx) if (len(text_idx) < self.max_len) else self.max_len)

        batch_max_len = max(batch_text_len)

        for text, label_text in zip(batch_text, batch_label_text):
            text_idx = self.tokenizer.encode(text, add_special_tokens=True)
            lt = self.tokenizer.encode(label_text, add_special_tokens=False)

            # 制作标签
            label_idx = [0] * len(text_idx)
            label_idx[self.mask_position[0] + 1] = lt[0]
            label_idx[self.mask_position[1] + 1] = lt[1]

            # PAD
            text_idx = text_idx + [0] * (batch_max_len - len(text_idx))
            label_idx = label_idx + [0] * (batch_max_len - len(label_idx))

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        return torch.tensor(batch_text_idx), torch.tensor(batch_label_idx), \
               torch.tensor(batch_label), torch.tensor(batch_text_len)

    def __len__(self):
        return len(self.all_label)

class PModel(torch.nn.Module):
    def __init__(self, model_name, prompt_num, tokenizer):
        super().__init__()
        self.bert1 = AutoModel.from_pretrained(model_name)
        self.bert1.resize_token_embeddings(len(tokenizer))#扩充词表后扩充bert embedding

        # for name, param in self.bert1.named_parameters():
        #     param.requires_grad = False

        self.generate_layer = torch.nn.Linear(768, 21128)

        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        attention_mask = (x != 103) & (x != 0)
        bert_out1, bert_out2 = self.bert1(x, return_dict=False, attention_mask=attention_mask)
        pre = self.generate_layer(bert_out1)

        if label is not None:
            loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)


def add_prompt_word(nums, tokenizer):
    prompt_tokens = []

    for i in range(nums):
        prompt_tokens.append(f"[prompt_{i}]")

    tokenizer.add_special_tokens({"additional_special_tokens": prompt_tokens})
    return tokenizer


if __name__ == "__main__":
    train_text, train_label = read_data(os.path.join("data", "train.txt"), 50)
    test_text, test_label = read_data(os.path.join("data", "test.txt"), 5000)

    with open(os.path.join("data", "index_2_label.txt"), encoding="utf-8") as f:
        index_2_label = f.read().split("\n")

    batch_size = 32
    epoch = 50
    lr = 1e-5
    prompt_num = 10
    max_len = 200
    mask_position = [0, 1]
    model_name = "./pretrained_model/bert-base-chinese"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = add_prompt_word(prompt_num, tokenizer)

    train_dataset = PDataset(train_text, train_label, tokenizer, index_2_label, prompt_num, mask_position, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=train_dataset.pro_batch_data)

    test_dataset = PDataset(test_text, test_label, tokenizer, index_2_label, prompt_num, mask_position, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=test_dataset.pro_batch_data)

    model = PModel(model_name, prompt_num, tokenizer).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        train_bar = tqdm(train_dataloader)

        for batch_text_idx, batch_label_idx, batch_class, batch_len in train_bar:
            train_bar.set_description("epoch:{}".format(e))

            batch_text_idx = batch_text_idx.to(device)
            batch_label_idx = batch_label_idx.to(device)

            loss = model.forward(batch_text_idx, batch_label_idx)
            loss.backward()

            opt.step()
            opt.zero_grad()

        model.eval()
        right = 0
        test_bar = tqdm(test_dataloader)
        for batch_text_idx, batch_label_idx, batch_class, batch_len in test_bar:
            test_bar.set_description("epoch:{}".format(e))

            batch_text_idx = batch_text_idx.to(device)
            batch_label_idx = batch_label_idx.to(device)

            pre = model.forward(batch_text_idx)

            for p, lab, type_, len_ in zip(pre, batch_label_idx, batch_class, batch_len):
                predict_text = tokenizer.decode(p[: len_][1:3])
                label_text = tokenizer.decode(lab[: len_][1:3])
                if (predict_text == label_text): right += 1

        acc = right / len(test_dataset)

        print(f"acc:{acc:.3f}")
