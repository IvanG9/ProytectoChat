import string

class Tokenizer:
    def __init__(self):
        self.all_characters = string.printable + "ñÑáÁéÉíÍóÓúÚ¿¡"
        self.n_characters = len(self.all_characters)

    def text_to_seq(self, text):
        return [self.all_characters.index(c) for c in text if c in self.all_characters]

    def seq_to_text(self, seq):
        return ''.join(self.all_characters[i] for i in seq)