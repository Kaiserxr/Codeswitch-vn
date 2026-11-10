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
- CanVEC sample (test set A): Nguyen & Bryant (2020), https://github.com/Bak3rLi/CanVEC. It is 99 utterances of real Vietnamese-English family conversation. Not included here: download `Sample_CanVEC_TEXT.txt` into `data/canvec/`, which is gitignored.
  - The sample has **no word-level language labels**, only a transcript. `python src/canvec.py prelabel` writes rule-based labels to `data/canvec/canvec_marked.txt`: a word with a tone mark is VI, a short list of unaccented Vietnamese words is VI, and everything else is EN. **These labels must be checked by hand before the scores count as official.** `python src/canvec.py eval` scores both models.
  - Label decisions: proper nouns (Jimmy, BTS, Sydney…) and fillers (mhm, uh, ah, oh) are marked `~` (OTHER) and not scored. `[A:…]` anonymisation tags and `X` (unintelligible) are dropped. Loanwords inside Vietnamese (concert, video) count as EN.

## Progress log
| Date | What changed |
|---|---|
| 2026-11-09 | Baseline tagger + accent-stripping revision |
| 2026-11-09 | CanVEC loader (`src/canvec.py`) and first test-set A run: baseline 95.8% accented / 83.8% no-accent; revised 89.2% / 88.7% (592 scored words). The revision helps no-accent text but hurts accented text: it now calls short English words (a, it, that, but, to) VI. |
| 2026-11-09 | Revision 2 (`revised2`): added a `common_en` feature (Oxford English Corpus top-100 list plus be/do/have forms) because the training data has almost no English function words. CanVEC: 94.8% accented / 93.9% no-accent. Internal split: 97.7% / 98.2%. It is now the default for tagging single sentences. Still wrong: that, than, lot, view, and loanwords inside Vietnamese (video, eye contact). |
