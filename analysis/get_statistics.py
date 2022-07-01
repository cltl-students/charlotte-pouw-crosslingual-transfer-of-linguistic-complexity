import os
import numpy as np

# meco_path = 'data/meco/files_per_language'
#
# for language_folder in os.listdir(meco_path):
#     if language_folder != 'Estonian' and language_folder != 'Italian':
#         with open(f'{meco_path}/{language_folder}/{language_folder.lower()}_clean.txt', encoding='utf8') as infile:

with open('data/geco/dutch/train_sentences.txt', 'r', encoding='utf8') as infile:

    train_sentences = infile.readlines()
    train_sentences = [s.strip('\n') for s in train_sentences]

    with open('data/geco/dutch/test_sentences.txt', 'r', encoding='utf8') as infile:

        # Retrieve the sentences
        test_sentences = infile.readlines()
        test_sentences = [s.strip('\n') for s in test_sentences]

        sentences = train_sentences + test_sentences

        splitted_sents = [sent.split(' ') for sent in sentences]

        sent_lengths = [len(sent) for sent in splitted_sents]
        word_lengths = [len(word) for sent in splitted_sents for word in sent]
        avg_sent_length = np.mean(sent_lengths)
        avg_word_length = np.mean(word_lengths)

       # print('Statistics for', language_folder, ':')
        print('Num sents:', len(sentences))
        print('Num words:', len(word_lengths))
        print('Avg sent length:', avg_sent_length)
        print('Avg word length', avg_word_length)
        print()