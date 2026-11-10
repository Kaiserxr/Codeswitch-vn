import re, random, unicodedata
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def load_marked(path):
    """Read sentences where English words start with '*'. Returns list of (tokens, labels)."""
    data = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks, labs = [], []
        for w in line.split():
            labs.append("EN" if w.startswith("*") else "VI")
            toks.append(w.lstrip("*"))
        data.append((toks, labs))
    return data


def strip_accents(text):
    """Remove Vietnamese tone marks: 'Hôm nay' -> 'Hom nay'. Also đ -> d."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# Revision 2: the 100 most frequent English words (Oxford English Corpus list),
# plus forms of be/do/have. Chosen from a general list, NOT from test-set errors.
# Some collide with Vietnamese typed without accents (me=mẹ, an=ăn, do=đó, the=thể,
# no=nó, can=cần); for those the model has to rely on the neighbouring words.
COMMON_EN = set("""
the be to of and a in that have i it for not on with he as you do at this but his by
from they we say her she or an will my one all would there their what so up out if about
who get which go me when make can like time no just him know take people into year your
good some could them see other than then now look only come its over think also back after
use two how our work first well way even new want because any these give day most us
is are was were am been being has had did does i'm it's don't
""".split())


def word_features(tokens, i, use_lexicon=False):
    """Describe word i using its letters and its neighbours."""
    w = tokens[i].lower()
    padded = f"<{w}>"
    f = {"word=" + w: 1, "has_vn_mark": int(strip_accents(w) != w)}
    if use_lexicon:
        f["common_en"] = int(w in COMMON_EN)
        f["prev_common_en"] = int(i > 0 and tokens[i - 1].lower() in COMMON_EN)
        f["next_common_en"] = int(i < len(tokens) - 1 and tokens[i + 1].lower() in COMMON_EN)
    for n in (1, 2, 3):
        for j in range(len(padded) - n + 1):
            f[f"ng{n}=" + padded[j:j + n]] = 1
    f["prev=" + (tokens[i - 1].lower() if i > 0 else "<START>")] = 1
    f["next=" + (tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>")] = 1
    return f


def to_xy(data, use_lexicon=False):
    X, y = [], []
    for toks, labs in data:
        for i in range(len(toks)):
            X.append(word_features(toks, i, use_lexicon))
            y.append(labs[i])
    return X, y


def train(data, use_lexicon=False):
    X, y = to_xy(data, use_lexicon)
    model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, C=5))
    model.fit(X, y)
    model.use_lexicon = use_lexicon   # remember which features this model expects
    return model


def features(model, tokens):
    return [word_features(tokens, i, getattr(model, "use_lexicon", False)) for i in range(len(tokens))]


def tag(model, sentence):
    toks = re.findall(r"[\wÀ-ỹ']+", sentence)
    preds = model.predict(features(model, toks))
    return list(zip(toks, preds))


def accuracy(model, data):
    X, y = to_xy(data, getattr(model, "use_lexicon", False))
    p = model.predict(X)
    scored = [(x, a, b) for x, a, b in zip(X, y, p) if a != "OTHER"]   # names/fillers are not scored
    correct = sum(a == b for _, a, b in scored)
    return correct / len(scored), [(x, a, b) for x, a, b in scored if a != b]


def no_accent_copy(data):
    return [([strip_accents(t) for t in toks], labs) for toks, labs in data]


if __name__ == "__main__":
    # Usage:
    #   python src/tagger.py                 -> train, then compare baseline vs revised on a held-out split
    #   python src/tagger.py "cau can tag"   -> train the revised model and tag your sentence
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    data = load_marked(os.path.join(here, "..", "data", "train_marked.txt"))
    if len(sys.argv) > 1:
        model = train(data + no_accent_copy(data), use_lexicon=True)
        print("  ".join(f"{w}/{lab}" for w, lab in tag(model, " ".join(sys.argv[1:]))))
    else:
        random.seed(0); random.shuffle(data)
        cut = int(len(data) * 0.8)
        tr, te = data[:cut], data[cut:]
        models = [("baseline", train(tr)),
                  ("revised", train(tr + no_accent_copy(tr))),
                  ("revised2", train(tr + no_accent_copy(tr), use_lexicon=True))]
        print(f"{'':10}{'accented':>10}{'no-accent':>11}")
        for name, m in models:
            print(f"{name:10}{accuracy(m, te)[0]:>10.1%}{accuracy(m, no_accent_copy(te))[0]:>11.1%}")
