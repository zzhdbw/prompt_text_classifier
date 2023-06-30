#硬模板prompt
import os
import torch
from transformers import BertModel, BertTokenizer
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

    assert len(all_text) == len(all_label)  # 文本和标签个数不一致就抛出异常

    if num is None:
        return all_text, all_label
    else:
        return all_text[:num], all_label[:num]

class PDataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, index2label, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.index2label = index2label
        self.max_len = max_len

        self.prompt_text1 = "文章内容是："
        # self.prompt_text1 = ""
        self.prompt_text2 = ",上文类别是："
    def __getitem__(self, index):
        text = self.all_text[index]
        label_id = self.all_label[index]
        text_label = self.index2label[label_id]

        text_prompt = self.prompt_text1 + text + self.prompt_text2 + "[MASK]" * 2

        return text_prompt, text_label, label_id

    def pro_batch_data(self, batch_data):
        text_prompts, text_labels, label_ids = zip(*batch_data)

        batch_text_idx = []
        batch_label_idx = []
        batch_text_len = []

        # 句长统计
        for text_prompt, text_label, label_id in zip(text_prompts, text_labels, label_ids):
            text_idx = self.tokenizer.encode(text_prompt, add_special_tokens=True)
            batch_text_len.append(len(text_idx) if (len(text_idx) < self.max_len) else self.max_len)

        batch_max_len = max(batch_text_len)

        for text_prompt, text_label, label_id in zip(text_prompts, text_labels, label_ids):
            text_idx = self.tokenizer.encode(text_prompt, add_special_tokens=True)
            label_idx = [self.tokenizer.pad_token_id] * (len(text_idx) - 3) + \
                        self.tokenizer.encode(text_label, add_special_tokens=False) + \
                        [self.tokenizer.pad_token_id]

            # 切割
            if (len(text_idx) > batch_max_len):
                text_idx = text_idx[:batch_max_len]
                label_idx = label_idx[:batch_max_len]

            # 填充
            if (len(text_idx) < batch_max_len):
                text_idx = text_idx + [self.tokenizer.pad_token_id] * (batch_max_len - len(text_idx))
                label_idx = label_idx + [self.tokenizer.pad_token_id] * (batch_max_len - len(label_idx))

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        return torch.tensor(batch_text_idx), torch.tensor(batch_label_idx), \
               torch.tensor(label_ids), torch.tensor(batch_text_len)

    def __len__(self):
        return len(self.all_label)

class PModel(torch.nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.bert1 = BertModel.from_pretrained(model_name)

        # for name, param in self.bert1.named_parameters():#冻结bert参数
        #     param.requires_grad = False

        self.tokenizer = tokenizer

        self.generate_layer = torch.nn.Linear(768, tokenizer.vocab_size)

        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def forward(self, x, label=None):
        attention_mask = (x != 103) & (x != 0)
        bert_out1, bert_out2 = self.bert1(x, return_dict=False, attention_mask=attention_mask)

        pre = self.generate_layer(bert_out1)

        if label is not None:
            loss = self.loss_fun(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == "__main__":
    train_text, train_label = read_data(os.path.join("data", "train.txt"), 50)
    test_text, test_label = read_data(os.path.join("data", "test.txt"), 5000)

    with open(os.path.join("data", "index_2_label.txt"), encoding="utf-8") as f:
        labels = f.read().split("\n")
    id2label = dict((index, label) for index, label in enumerate(labels))
    label2id = dict((label, index) for index, label in enumerate(labels))

    batch_size = 32
    epoch = 50
    max_len = 200
    lr = 1e-5

    model_name = "./pretrained_model/bert-base-chinese"

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = PDataset(train_text, train_label, tokenizer, id2label, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.pro_batch_data)

    test_dataset = PDataset(test_text, test_label, tokenizer, id2label, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=test_dataset.pro_batch_data)

    model = PModel(model_name, tokenizer).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        train_bar = tqdm(train_dataloader)
        for (batch_text_idx, batch_label_idx, label_ids, batch_text_len) in train_bar:
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
        for (batch_text_idx, batch_label_idx, label_ids, batch_text_len) in test_bar:
            test_bar.set_description("epoch:{}".format(e))
            batch_text_idx = batch_text_idx.to(device)
            batch_label_idx = batch_label_idx.to(device)

            predict = model.forward(batch_text_idx)

            for p, lab, len_ in zip(predict, batch_label_idx, batch_text_len):
                predict_text = tokenizer.decode(p[: len_][-3:-1])
                label_text = tokenizer.decode(lab[: len_][-3:-1])
                if (predict_text == label_text): right += 1

        acc = right / len(test_dataset)
        print(f"acc:{acc:.3f}")
