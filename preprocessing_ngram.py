import nltk
from nltk.corpus import stopwords, treebank
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter

nltk.download('stopwords')
nltk.download('treebank')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

sentences = treebank.sents()

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default noun

normalized_sentences = []

for sent in sentences[:10]:  
    tokens_no_stop = [w for w in sent if w.lower() not in stop_words]
    tokens_clean = [w for w in tokens_no_stop if w.isalpha()]
    pos_tags = nltk.pos_tag(tokens_clean)
    tokens_lemmatized = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag)) for word, tag in pos_tags]
    normalized_sentences.append(tokens_lemmatized)

print("\n Sentences After Pre processing:- \n",normalized_sentences)

# ---- N-Grams ----
n = 2  
all_tokens = [token for sent in normalized_sentences for token in sent]

n_grams = list(ngrams(all_tokens, n))
n_gram_freq = Counter(n_grams)
unigram_freq = Counter(all_tokens)

print("\n=== Sentence Probabilities (Markov Assumption) ===\n")

for i, sent in enumerate(normalized_sentences):
    sent_ngrams = list(ngrams(sent, n))
    sentence_prob = 1.0

    print(f"\nSentence {i+1}: {' '.join(sent)}")
    print("Bigram details:")

    for bg in sent_ngrams:
        prev_word = bg[0]
        next_word = bg[1]
        count_bigram = n_gram_freq[bg]
        count_unigram = unigram_freq[prev_word]

        # Laplace smoothing
        prob = (count_bigram + 1) / (count_unigram + len(unigram_freq))
        
        print(f"  {bg} -> count={count_bigram}, prev_count={count_unigram}, prob={prob:.6f}")

        sentence_prob *= prob

    print(f"➡️ Sentence {i+1} Probability = {sentence_prob:.12f}\n")

print("=== All Tokens Combined from 10 Sentences ===")
print(all_tokens)
print("\nTotal Tokens:", len(all_tokens))
