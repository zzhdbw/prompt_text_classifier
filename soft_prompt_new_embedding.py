#软模板prompt使用额外embedding编码模板
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
    def __init__(self, all_text, all_label, tokenizer, index_2_label, prompt_num, mask_position):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.index_2_label = index_2_label

        self.prompt_len = prompt_num
        self.mask_position = mask_position
        self.prompt_content = self.generate_template()

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

        # text_prompt = self.prompt_text1 + text + self.prompt_text2
        # text_prompt = ["[prompt_0][prompt_1][prompt_2][MASK][prompt_3][MASK][prompt_4][prompt_5][prompt_99]国足获得了世界杯冠军"]
        # label_prompt = [0, 0 ,0,1000,0,1001,0,0]
        text_prompt = self.prompt_content + text
        return text_prompt, label, label_text, len(text) + self.prompt_len + 2

    # ('[MASK][prompt_0][MASK][prompt_1][prompt_2][prompt_3][prompt_4][prompt_5][prompt_6][prompt_7][prompt_8][prompt_9]中华女子学院：本科层次仅1专业招男生',3, '教育', 30)
    def pro_batch_data(self, batch_data):
        batch_text, batch_label, batch_label_text, batch_len = zip(*batch_data)

        batch_max_len = max(batch_len) + 2

        batch_text_idx = []
        batch_label_idx = []
        batch_text_len = []

        for text, label, label_text, len_ in zip(batch_text, batch_label, batch_label_text, batch_len):
            text_idx = self.tokenizer.encode(text, add_special_tokens=True)
            lt = self.tokenizer.encode(label_text, add_special_tokens=False)

            label_idx = [0] * len(text_idx)
            label_idx[self.mask_position[0] + 1] = lt[0]
            label_idx[self.mask_position[1] + 1] = lt[1]

            batch_text_len.append(len(text_idx) - 2)

            assert len(text_idx) == len(label_idx)

            # PAD
            text_idx = text_idx + [0] * (batch_max_len - len(text_idx))
            label_idx = label_idx + [0] * (batch_max_len - len(label_idx))

            assert len(text_idx) == len(label_idx)

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        return torch.tensor(batch_text_idx), torch.tensor(batch_label_idx), \
               torch.tensor(batch_label), torch.tensor(batch_text_len)

    def __len__(self):
        return len(self.all_label)

class Prompt_Embedding(torch.nn.Module):
    def __init__(self, prompt_num):
        super().__init__()
        self.embedding = torch.nn.Embedding(prompt_num, 768)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.GELU()
        )

    def forward(self, x):
        emb = self.embedding(x)
        x = self.linear(emb)
        return x

class PModel(torch.nn.Module):
    def __init__(self, model_name, prompt_num):
        super().__init__()
        self.bert1 = AutoModel.from_pretrained(model_name)

        # for name, param in self.bert1.named_parameters():
        #     param.requires_grad = False

        self.prompt_embedding = Prompt_Embedding(prompt_num)
        self.bert_embedding = self.bert1.get_input_embeddings()

        self.generate_layer = torch.nn.Linear(768, 21128)

        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        x_copy = copy.deepcopy(x)
        prompt_position = x_copy >= 21128

        x[x < 21128] = 0
        x_copy[x_copy >= 21128] = 0

        prompt_emb = self.prompt_embedding(x[x > 0] - 21128)
        bert_emb = self.bert_embedding(x_copy)

        bert_emb[prompt_position] = prompt_emb

        bert_out1, bert_out2 = self.bert1(inputs_embeds=bert_emb, return_dict=False)

        pre = self.generate_layer(bert_out1)

        if label is not None:
            loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)
            # return pre

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

    assert len(train_text) == len(train_label)
    assert len(test_text) == len(test_label)

    batch_size = 32
    epoch = 50
    lr = 1e-5
    model_name = "./pretrained_model/bert-base-chinese"

    prompt_num = 10
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = add_prompt_word(prompt_num, tokenizer)
    mask_position = [0, 1]

    train_dataset = PDataset(train_text, train_label, tokenizer, index_2_label, prompt_num, mask_position)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=train_dataset.pro_batch_data)

    test_dataset = PDataset(test_text, test_label, tokenizer, index_2_label, prompt_num, mask_position)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=test_dataset.pro_batch_data)

    model = PModel(model_name, prompt_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_bar = tqdm(train_dataloader)
    for e in range(epoch):
        model.train()
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
