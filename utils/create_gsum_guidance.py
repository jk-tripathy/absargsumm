import re
from collections import OrderedDict

import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS


class GSumOracleSummary:
    def _get_ngrams(self, n, text):
        """Calculates n-grams.

        Args:
          n: which n-grams to calculate
          text: An array of tokens
        Returns:
          A set of n-grams
        """
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _get_word_ngrams(self, n, sentences):
        """Calculates word n-grams for multiple sentences."""
        assert len(sentences) > 0
        assert n > 0
        words = sum(sentences, [])
        return self._get_ngrams(n, words)

    def _cal_rouge(self, evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    def _greedy_selection(self, doc_sent_list, abstract_sent_list, summary_size):
        def _rouge_clean(s):
            return re.sub(r"[^a-zA-Z0-9 ]", "", s)

        max_rouge = 0.0
        abstract = abstract_sent_list
        abstract = _rouge_clean(" ".join(abstract)).split()
        sents = [_rouge_clean(s).split() for s in doc_sent_list]
        evaluated_1grams = [self._get_word_ngrams(1, [sent]) for sent in sents]
        reference_1grams = self._get_word_ngrams(1, [abstract])
        evaluated_2grams = [self._get_word_ngrams(2, [sent]) for sent in sents]
        reference_2grams = self._get_word_ngrams(2, [abstract])

        selected = []
        for s in range(summary_size):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(sents)):
                if i in selected:
                    continue
                c = selected + [i]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                candidates_2 = set.union(*map(set, candidates_2))
                rouge_1 = self._cal_rouge(candidates_1, reference_1grams)["f"]
                rouge_2 = self._cal_rouge(candidates_2, reference_2grams)["f"]
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if cur_id == -1:
                return sorted(selected)
            selected.append(cur_id)
            max_rouge = cur_max_rouge

        return sorted(selected)

    def get_oracle(
        self,
        src: str,
        tgt: str,
    ) -> str:
        src = src.rstrip().split("<q>")
        tgt = tgt.rstrip().split("<q>")
        sels = self._greedy_selection(src, tgt, 3)
        temps = [src[i] for i in sels]
        formatted_temp = "<q>".join(temps)
        return formatted_temp


class GSumGuidance:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.oracle = GSumOracleSummary()
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def set_stopwords(self, stopwords):
        """Set stop words."""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos."""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with candidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens."""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences."""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix."""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype="float")
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm
        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords."""
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True)
        )
        words = []
        for i, (key, value) in enumerate(node_weight.items()):
            words.append(key)
            if i > number:
                break
        return words

    def analyze(
        self,
        text,
        candidate_pos=["NOUN", "PROPN"],
        window_size=4,
        lower=False,
        stopwords=list(),
    ):
        """Main function to analyze text."""
        # Set stop words
        self.set_stopwords(stopwords)
        # Pare text by spaCy
        doc = self.nlp(text)
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        self.node_weight = node_weight

    def format_text(self, src, tgt):
        """Format with special <q> token."""
        src_sentences = sent_tokenize(src)
        src = "<q>".join(src_sentences)
        tgt_sentences = sent_tokenize(tgt)
        tgt = "<q>".join(tgt_sentences)
        return src, tgt

    def get_guidance(
        self,
        src: str,
        tgt: str,
        sep_token: str = "[SEP]",
        add_special_token: bool = True,
    ) -> str:
        formatted_src, formatted_tgt = self.format_text(src, tgt)
        oracle_summary = self.oracle.get_oracle(formatted_src, formatted_tgt)

        oracle_summary = src.strip().replace("<q>", " ")
        self.analyze(
            oracle_summary,
            candidate_pos=["NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"],
            window_size=4,
            lower=True,
        )
        words = self.get_keywords(50)
        tgt = tgt.strip().replace("<q>", " ")
        new_words = []
        for word in words:
            flag = True
            ws = word.split()
            for w in ws:
                if w not in tgt:
                    flag = False
                    break
            if flag:
                new_words.append(word)
        output = f" {sep_token} ".join(new_words)
        return output
