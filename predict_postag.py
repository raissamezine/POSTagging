import os
import subprocess
import torch
import torch.nn as nn
import lib.conllulib as conllulib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("device:", device)


class POSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim, hid_dim, pad_word_id, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_word_id)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        out = self.dropout(out)
        return self.fc(out)


def predict_file(model, Vw, id2tag, conllu_in, conllu_out, unk_id, device):
    model.eval()

    with open(conllu_in, encoding="utf-8") as f_in, open(conllu_out, "w", encoding="utf-8") as f_out:
        reader = conllulib.CoNLLUReader(f_in)

        for sent in reader.readConllu():
            words = [tok["form"] for tok in sent]
            x_ids = [Vw.get(w, unk_id) for w in words]
            x = torch.tensor([x_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(x)

            preds = logits.argmax(dim=-1)[0].tolist()

            for i, tok in enumerate(sent):
                tok["upos"] = id2tag[preds[i]]

            f_out.write(sent.serialize())
            f_out.write("\n")




checkpoint = torch.load("resultats/postagger.pt", map_location=device)

Vw = checkpoint["Vw"]
Vt = checkpoint["Vt"]

pad_word_id = checkpoint.get("pad_word_id", checkpoint.get("PAD_WORD_ID", Vw.get("<PAD>", 0)))
unk_id = checkpoint.get("unk_id", checkpoint.get("UNK_ID", Vw.get("<UNK>", 1)))
emb_dim = checkpoint.get("emb_dim", checkpoint.get("EMB_DIM", 100))
hid_dim = checkpoint.get("hid_dim", checkpoint.get("HID_DIM", 128))
dropout = checkpoint.get("dropout", 0.2)

state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint.get("model_state", None)))

model = POSTagger(
    vocab_size=len(Vw),
    tagset_size=len(Vt),
    emb_dim=emb_dim,
    hid_dim=hid_dim,
    pad_word_id=pad_word_id,
    dropout=dropout,
).to(device)

if state is None:
    raise KeyError("Checkpoint missing model_state_dict")

model.load_state_dict(state)
model.eval()

id2tag = conllulib.Util.rev_vocab(Vt)

predict_file(
    model=model,
    Vw=Vw,
    id2tag=id2tag,
    conllu_in="sequoia/sequoia-ud.parseme.frsemcor.simple.dev",
    conllu_out="resultats/sequoia-ud.parseme.frsemcor.simple-rnn-dev.pred",
    unk_id=unk_id,
    device=device,
)


