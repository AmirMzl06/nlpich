import pickle
import numpy as np
import re
import matplotlib.pyplot as plt

def remove_punctuation(sentence):
    # Remove punctuation


    trueSent = sentence.replace('>', ' ')
    trueSent = trueSent.replace('~', '.')
    trueSent = trueSent.replace('#', '')
    sentence = trueSent.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence


def compute_wer(r, h):
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]



def _cer_and_wer(decodedSentences, trueSentences,):
    allCharErr = []
    allChar = []
    allWordErr = []
    allWord = []
    cer_list, wer_list = [], []
    for x in range(len(decodedSentences)):
        decSent = decodedSentences[x]
        trueSent = trueSentences[x]

        trueSent = remove_punctuation(trueSent)
        decSent = remove_punctuation(decSent)
        trueWords = trueSent.replace(">", " > ").split(" ")
        decWords = decSent.replace(">", " > ").split(" ")
        nCharErr = compute_wer([c for c in trueSent], [c for c in decSent])
        nWordErr = compute_wer(trueWords, decWords)
        # print(trueSent)
        allCharErr.append(nCharErr)
        allWordErr.append(nWordErr)
        cer_list.append(nCharErr / len(trueSent))
        wer_list.append(nWordErr / len(trueWords))
        allChar.append(len(trueSent))
        allWord.append(len(trueWords))

    cer = np.sum(allCharErr) / np.sum(allChar)
    wer = np.sum(allWordErr) / np.sum(allWord)

    return cer, wer, cer_list, wer_list

