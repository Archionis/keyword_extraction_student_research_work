from sklearn.feature_extraction.text import TfidfVectorizer

class CustomVectorizer(TfidfVectorizer): 
    def __init__(self, nlp_model=None, allowed_patterns=None, 
                 ngram_range=(1, 1), max_features=None, min_df=1, 
                 max_df=1.0, stop_words=None, lowercase=True):
        super().__init__(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            token_pattern=r'(?u)\b\w\w+\b'
        )
        self.nlp_model = nlp_model
        self.allowed_patterns = allowed_patterns or [
            ('NOUN',), ('PROPN',),
            ('ADJ', 'NOUN'), ('ADJ', 'PROPN'),
            ('NOUN', 'NOUN'), ('NOUN', 'PROPN'),
            ('PROPN', 'NOUN'), ('PROPN', 'PROPN'),
            ('ADJ', 'ADJ', 'NOUN'),
            ('ADJ', 'NOUN', 'NOUN'),
            ('NOUN', 'NOUN', 'NOUN'),
            ('ADJ', 'NOUN', 'PROPN'),
            ('NOUN', 'ADJ', 'NOUN'),
            ('NOUN', 'NOUN', 'PROPN'),
        ]
    def build_analyzer(self):
        default_analyzer = super().build_analyzer()
        if self.nlp_model is None:
            return default_analyzer
        def analyzer(doc):
            if self.lowercase:
                doc = doc.lower()
            spacy_doc = self.nlp_model(doc)
            tokens = [(token.text, token.pos_) for token in spacy_doc]
            candidates = []
            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_tokens = tokens[i:i+n]
                    ngram_text = " ".join([t[0] for t in ngram_tokens])
                    ngram_pos = tuple([t[1] for t in ngram_tokens])
                    if ngram_pos in self.allowed_patterns:
                        candidates.append(ngram_text)
            if candidates:
                return candidates
            else:
                return default_analyzer(doc)
        return analyzer
