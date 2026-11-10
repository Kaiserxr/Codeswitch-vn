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


def word_features(tokens, i):
    """Describe word i using its letters and its neighbours."""
    w = tokens[i].lower()
    padded = f"<{w}>"
    f = {"word=" + w: 1, "has_vn_mark": int(strip_accents(w) != w)}
    for n in (1, 2, 3):
        for j in range(len(padded) - n + 1):
            f[f"ng{n}=" + padded[j:j + n]] = 1
    f["prev=" + (tokens[i - 1].lower() if i > 0 else "<START>")] = 1
    f["next=" + (tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>")] = 1
    return f


def to_xy(data):
    X, y = [], []
    for toks, labs in data:
        for i in range(len(toks)):
            X.append(word_features(toks, i))
            y.append(labs[i])
    return X, y


def train(data):
    X, y = to_xy(data)
    model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=2000, C=5))
    model.fit(X, y)
    return model


def tag(model, sentence):
    toks = re.findall(r"[\wÀ-ỹ']+", sentence)
    preds = model.predict([word_features(toks, i) for i in range(len(toks))])
    return list(zip(toks, preds))


def accuracy(model, data):
    X, y = to_xy(data)
    p = model.predict(X)
    correct = sum(a == b for a, b in zip(p, y))
    return correct / len(y), [(x, a, b) for x, a, b in zip(X, y, p) if a != b]


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
        model = train(data + no_accent_copy(data))
        print("  ".join(f"{w}/{lab}" for w, lab in tag(model, " ".join(sys.argv[1:]))))
    else:
        random.seed(0); random.shuffle(data)
        cut = int(len(data) * 0.8)
        tr, te = data[:cut], data[cut:]
        base, fixed = train(tr), train(tr + no_accent_copy(tr))
        print(f"{'':10}{'accented':>10}{'no-accent':>11}")
        for name, m in [("baseline", base), ("revised", fixed)]:
            print(f"{name:10}{accuracy(m, te)[0]:>10.1%}{accuracy(m, no_accent_copy(te))[0]:>11.1%}")
