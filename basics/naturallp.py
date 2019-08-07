'''

'''

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

'''
Creates a dictionary for each word in the text
'''
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

'''
Encodes the sentences into seqence of encodings that correspond to the dictionary.
'''
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen = 8)
print("\nWord Index = ", word_index)
print("\nSequences = ", sequences)
print("\nPadded Sequences:")
print(padded)

'''
Testing
'''
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequences = ", test_sequences)

padded = pad_sequences(test_sequences, maxlen = 10)
print("\nPadded Test Sequence:")
print(padded)




