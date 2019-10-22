"""
A greedy algorithmic implementation to generate a corpus of phonetically balanced speaker lists,
where each speaker list consists of a unique set of Hangul sentences (Korean).
"""

import random
import ast
import numpy
from tqdm import tqdm


# a comprehensive list of phones for korean
# these were extracted from KoG2P's github repo at https://github.com/scarletcho/KoG2P
PHONES = ['p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr',
          'pf', 'ph', 'tf', 'th', 'kf', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'mf', 'nf', 'ng', 'll', 'ks', 'nc',
          'nh', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo', 'ye',
          'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we', 'wa', 'wv', 'xi']


def vectorize(fphones, fvectors):
    """
    Vectorizes `fphones` by computing the per-phoneme frequency occurrences in a given sentence and
    appends a unique utterance id for each vector in `fphones`
    :param fphones: File path (str) to phonetic representations of sentences to select from
    :param fvectors: File path (str) to save transformed `fphones` to
    """
    print('Vectorizing {} to {}...'.format(fphones, fvectors))
    with open(fphones, 'r') as fin, open(fvectors, 'w') as fout:
        sent_id = 0
        for sent in fin.readlines():
            sent = sent.strip().split()
            sent_vec = [0 for _ in PHONES]
            for phone in sent:
                sent_vec[PHONES.index(phone)] += 1
            sent_vec.append(sent_id)
            sent_id += 1
            fout.write('{}\n'.format(str(sent_vec)).replace('[', '').replace(']', '').replace(',', ''))


def compute_normalized_dot_product(vec_a, vec_b):
    """
    Computes the normalized dot product as a similarity measure of `vec_a` and `vec_b`
    :param vec_a: An array (a list or a numpy array)
    :param vec_b: An array (a list or a numpy array)
    :return: The normalized dot product of vec_a and vec_b
    """
    return numpy.dot(vec_a, vec_b) / (numpy.linalg.norm(vec_a) * numpy.linalg.norm(vec_b))


def generate_speaker_lists(num_sents_in_pool, num_sents_per_spkr, num_speakers, num_sents_per_partition, target_vec):
    """
    Generates phonetically balanced speaker lists in a greedy fashion such that each speaker list
    individually resembles the global phonetic distribution of all sentences in our pool. This is
    implemented by specifying the target phonetic distribution for each speaker list (see `target_vec`)
    :param num_sents_in_pool: Size of our entire sentence pool to select over (|S|)
    :param num_sents_per_spkr: The length of each speaker list (K)
    :param num_speakers: The number of speaker lists (N)
    :param num_sents_per_partition: Size of the subset of candidate sentences in each iteration. This
     is a heuristic to speed up generation that should be fine-tuned (P)
    :param target_vec: The target phonetic distribution for each speaker list
    :return: Saves the generated speaker lists in `speakers.txt` with a newline marking the next speaker. Saves
    the vector similarity (normalized dot product) of each generated speaker list with `target_vec` in `log.txt`
    """
    print('Generating speaker lists...')
    with \
            open('vectors-all.txt', 'r') as fin, \
            open('hangul-all.txt', 'r') as fhan, \
            open('speakers.txt', 'w') as fout, \
            open('log.txt', 'w') as flog:

        # load hangul sentences
        hangul = fhan.readlines()

        # load and shuffle vectors (vectorized phonemes)
        vectors = fin.readlines()
        numpy.random.shuffle(vectors)

        used_sents = set()
        for _ in tqdm(range(num_speakers)):
            # build this speaker's list
            speaker_sents = []
            speaker_dist = numpy.zeros(shape=len(PHONES), dtype=int)

            while len(speaker_sents) < num_sents_per_spkr:
                # load random subset of vectors
                selected_lines = [vectors[random.randrange(0, num_sents_in_pool)] for _ in range(num_sents_per_partition)]
                subset = []
                for vector in selected_lines:
                    subset.append([int(x) for x in vector.strip().split(' ')])

                # find the best vector in the subset
                max_phonetic_dist = 0
                best_candidate_sent = None
                best_candidate_vec = None

                for i in range(len(subset)):

                    # skip if sentence has already been used
                    if subset[i][66] in used_sents:
                        continue

                    # else get stats from the sentence
                    updated_speaker_dist = speaker_dist + subset[i][:66]
                    phonetic_dist = compute_normalized_dot_product(updated_speaker_dist, target_vec)

                    # and update running stats for this subset
                    if phonetic_dist > max_phonetic_dist:
                        max_phonetic_dist = phonetic_dist
                        best_candidate_sent = subset[i][66]
                        best_candidate_vec = subset[i][:66]

                # select the best sentence from this subset and assign to current speaker
                if best_candidate_sent:
                    speaker_sents.append(best_candidate_sent)
                    speaker_dist += best_candidate_vec
                    used_sents.add(best_candidate_sent)
                else:
                    continue

            # log each speaker's overall similarity with the per-speaker target distribution
            flog.write('{}\n'.format(compute_normalized_dot_product(speaker_dist, target_vec)))

            # retrieve corresponding hangul sentences and save it
            for sent_id in speaker_sents:
                fout.write('{}'.format(hangul[sent_id]))
            # use newline in saved file to mark a new speaker
            fout.write('\n')


def main():
    """
    This is the entry point of this code where we set some key constants
    """
    # we set some known constants
    num_sents_in_pool = 22051730
    num_sents_per_spkr = 160
    num_speakers = 260
    num_sents_per_partition = 10  # heuristic to speed up algorithm

    # run only once to generate the corresponding vectors file for a given phones file
    # this is a slow process so we save the generated vectors to `vectors-all.txt` for further use
    # vectorize('phones-all.txt', 'vectors-all.txt')

    # load known global phonetic distribution
    with open('global-phone-dist.txt', 'r') as fin:
        global_phones = ast.literal_eval(fin.readline().strip())

    # find per-speaker target phonetic distribution
    target_dist = {phone: int((global_phones[phone] * num_sents_per_spkr) / num_sents_in_pool) for phone in global_phones}
    target_vec = numpy.array([target_dist[phone] for phone in PHONES])
    print('Per-speaker target distribution vector is: {}'.format(target_vec))

    # actually generate the speaker lists
    generate_speaker_lists(num_sents_in_pool, num_sents_per_spkr, num_speakers, num_sents_per_partition, target_vec)


if __name__ == "__main__":
    main()
