"""CanVEC sample -> test set A.

The public CanVEC sample (Nguyen & Bryant 2020, https://github.com/Bak3rLi/CanVEC)
is a plain transcript: one tab-separated line per utterance, with columns
Speaker, Generation, Transcript, Time start, Time finish, IU. It has NO word-level
language labels, so we make our own in two steps:

  1. python src/canvec.py prelabel
       Reads data/canvec/Sample_CanVEC_TEXT.txt and writes
       data/canvec/canvec_marked.txt in the same format as train_marked.txt,
       using simple rules (see auto_label). A rule-based pre-label is NOT gold:
       a native speaker must read every line and fix it by hand.
  2. python src/canvec.py eval
       Scores baseline and revised models on the hand-checked file,
       both as written (accented) and with tone marks stripped.

Label decisions (also in README):
  *word  -> EN
  word   -> VI
  ~word  -> OTHER: kept as context for the model, but NOT scored.
            Used for proper nouns (Jimmy, BTS, Sydney...) and fillers (mhm, uh, ah, oh),
            because neither VI nor EN is clearly "right" for them.
  Dropped entirely: [A:...] anonymisation tags and X / <X> (unintelligible speech).
  Loanwords used inside Vietnamese (concert, video) are labelled EN: they are
  English words a parent would want to spot.

The data is third-party: data/canvec/ is gitignored, do not commit it.
"""
import os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tagger import strip_accents, load_marked, train, no_accent_copy, features

HERE = os.path.dirname(os.path.abspath(__file__))
CANVEC_DIR = os.path.join(HERE, "..", "data", "canvec")
RAW = os.path.join(CANVEC_DIR, "Sample_CanVEC_TEXT.txt")
MARKED = os.path.join(CANVEC_DIR, "canvec_marked.txt")
TRAIN = os.path.join(HERE, "..", "data", "train_marked.txt")

# Vietnamese words in the sample that happen to have no tone mark.
# Without this list, "no tone mark" would wrongly mean "English".
VI_PLAIN = {"ai", "ba", "cao", "cho", "coi", "con", "hai", "hay", "nghe", "nhau", "sao", "trong"}
PROPER = {"jimmy", "bts", "leonardo", "dicaprio", "sydney", "instagram", "kpop"}
FILLERS = {"mhm", "uh", "um", "ah", "oh"}


def read_utterances(path=RAW):
    """Return the IU (utterance text) column as a list of strings."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = header.index("IU")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > col:
                out.append(unicodedata.normalize("NFC", parts[col]).strip())
    return out


def tokenize(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)   # [A:person name]
    text = re.sub(r"<[^>]*>", " ", text)      # <X>
    toks = re.findall(r"[\wÀ-ỹ']+", text)
    return [t for t in toks if t != "X"]      # X = unintelligible


def auto_label(word):
    w = word.lower()
    if w in PROPER or w in FILLERS:
        return "OTHER"
    if strip_accents(w) != w or w in VI_PLAIN:
        return "VI"
    return "EN"


def prelabel():
    if os.path.exists(MARKED) and "--force" not in sys.argv:
        sys.exit(f"{MARKED} already exists (maybe hand-edited). Use --force to overwrite.")
    lines = ["# CanVEC sample, auto pre-labelled by src/canvec.py. CHECK EVERY LINE BY HAND.",
             "# *word = EN, plain = VI, ~word = OTHER (name/filler, not scored)."]
    for utt in read_utterances():
        toks = tokenize(utt)
        if not toks:
            continue
        mark = {"EN": "*", "VI": "", "OTHER": "~"}
        lines.append(" ".join(mark[auto_label(t)] + t for t in toks))
    with open(MARKED, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 2} sentences to {MARKED}")


def load_canvec_marked(path=MARKED):
    """Like load_marked, but ~word gets label OTHER."""
    data = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks, labs = [], []
        for w in line.split():
            if w.startswith("*"):
                labs.append("EN")
            elif w.startswith("~"):
                labs.append("OTHER")
            else:
                labs.append("VI")
            toks.append(w.lstrip("*~"))
        data.append((toks, labs))
    return data


def score(model, data):
    """Accuracy over VI/EN words only (OTHER is skipped). Returns (acc, n, errors)."""
    right, n, errors = 0, 0, []
    for toks, labs in data:
        preds = model.predict(features(model, toks))
        for i, (gold, p) in enumerate(zip(labs, preds)):
            if gold == "OTHER":
                continue
            n += 1
            if gold == p:
                right += 1
            else:
                errors.append((" ".join(toks), toks[i], gold, p))
    return right / n, n, errors


def evaluate(show_errors=True):
    train_data = load_marked(TRAIN)
    test = load_canvec_marked()
    mixed = [s for s in test if {"VI", "EN"} <= set(s[1])]
    models = [("baseline", train(train_data)),
              ("revised", train(train_data + no_accent_copy(train_data))),
              ("revised2", train(train_data + no_accent_copy(train_data), use_lexicon=True))]
    print(f"CanVEC: {len(test)} sentences ({len(mixed)} contain both VI and EN)")
    print(f"{'':10}{'accented':>10}{'no-accent':>11}{'mixed only':>12}")
    for name, m in models:
        a, n, _ = score(m, test)
        b, _, _ = score(m, no_accent_copy(test))
        c, _, _ = score(m, mixed)
        print(f"{name:10}{a:>10.1%}{b:>11.1%}{c:>12.1%}   (n={n} scored words)")
    if show_errors:
        for name, m in models[2:]:
            for label, d in [("accented", test), ("no-accent", no_accent_copy(test))]:
                _, _, errs = score(m, d)
                print(f"\n{name} / {label}: {len(errs)} errors")
                for sent, w, gold, p in errs:
                    print(f"  {w!r:16} gold={gold} pred={p}   | {sent}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "prelabel":
        prelabel()
    elif cmd == "eval":
        evaluate(show_errors="--quiet" not in sys.argv)
    else:
        sys.exit(__doc__)
