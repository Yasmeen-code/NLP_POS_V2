import nltk
import re
from nltk.corpus import stopwords, treebank
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.util import bigrams

# ======= Load Treebank Dataset =======
sentences = [" ".join(sentence) for sentence in treebank.sents()]
corpus = sentences[:10] 

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# ======= POS Mapping =======
def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    elif tag.startswith('V'): return 'v'
    elif tag.startswith('N'): return 'n'
    elif tag.startswith('R'): return 'r'
    return 'n'
# ======= Text Normalization =======
def normalize_text(text_list):
    cleaned = []
    for sent in text_list:
        text = sent.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [w for w in tokens if w.isalpha()]

        pos_tags = nltk.pos_tag(tokens)

        lemmas = [
            lemmatizer.lemmatize(w, get_wordnet_pos(tag))
            for w, tag in pos_tags
        ]

        cleaned.append(lemmas)
    return cleaned
# ===== Normalize Dataset =====
normalized_sentences = normalize_text(corpus)

print("="*60)
print(" PART 1 — Sentences After Pre-processing ")
print("="*60)
for i, sent in enumerate(normalized_sentences, 1):
    print(f"S{i} (Original): {corpus[i-1]}")
    print(f"S{i} (Cleaned): {sent}")
    print("-" * 30)

# ===== Build Bigram Model =====
bigram_sentences = [['<s>'] + sent for sent in normalized_sentences]
all_tokens = [tok for sent in bigram_sentences for tok in sent]

unigram_counts = Counter(all_tokens)
vocab_size = len(unigram_counts)

all_bigrams = []
for sent in bigram_sentences:
    all_bigrams.extend(list(bigrams(sent)))

bigram_counts = Counter(all_bigrams)

# ===== Bigram Probability Function =====
def get_bigram_probability(w_prev, w_curr):
    count_bg = bigram_counts.get((w_prev, w_curr), 0)
    count_prev = unigram_counts.get(w_prev, 0)
    return (count_bg + 1) / (count_prev + vocab_size)

# ===== Sentence Probability =====
def calculate_sentence_probability(tokens):
    full_sent = ['<s>'] + tokens
    sentence_bigrams = list(bigrams(full_sent))

    prob_product = 1.0
    steps = []

    for w_prev, w_curr in sentence_bigrams:
        prob = get_bigram_probability(w_prev, w_curr)
        prob_product *= prob

        steps.append(
            f"P({w_curr} | {w_prev}) = "
            f"({bigram_counts.get((w_prev,w_curr),0)} + 1) / "
            f"({unigram_counts.get(w_prev,0)} + {vocab_size}) = {prob:.6f}"
        )

    return prob_product, steps

# ===== PRINT RESULTS =====
print("\n" + "="*60)
print(" PART 2 — Bigram Probabilities for 10 Sentences ")
print(" Using Laplace Smoothing | V =", vocab_size)
print("="*60)

for i, sent in enumerate(normalized_sentences, 1):

    if not sent:
        print(f"\nSentence {i}: EMPTY → Probability = 0.0")
        continue

    prob, steps = calculate_sentence_probability(sent)

    print(f"\nSentence {i}: {' '.join(sent)}")
    print("Bigram Details:")

    for st in steps:
        print("  -", st)

    print(f"➡ Final Probability = {prob:.18f}")
