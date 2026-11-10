# Vietnamese–English Code-Switching Tagger

CAI 1000 Capstone Project. Labels every word in a mixed Vietnamese-English sentence as **VI** or **EN**.

```
Input:  hom nay minh co deadline nen phai rush lam
Output: hom/VI nay/VI minh/VI co/VI deadline/EN nen/VI phai/VI rush/EN lam/VI
```

## Run it
```bash
pip install -r requirements.txt
python src/tagger.py                              # baseline vs revised accuracy
python src/tagger.py "Tối nay đi ăn không mình hơi bored"   # tag a sentence
jupyter notebook notebooks/codeswitch_tagger.ipynb          # full walkthrough
```

## Data
- `data/train_marked.txt`: 155 training sentences. Words starting with `*` are English. Drafted with Claude (generative AI) and reviewed by the author.
- CanVEC sample (test set): Nguyen & Bryant (2020), https://github.com/Bak3rLi/CanVEC. Not included here; put it in `data/canvec/`.

## Progress log
| Date | What changed |
|---|---|
| 2026-11-09 | Baseline tagger + accent-stripping revision |
