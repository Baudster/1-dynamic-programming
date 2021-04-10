#!/usr/bin/env python3

import numpy as np

# %%
# Assignment Pt. 1: Edit Distances

gem_doppel = [
    ("GCGTATGAGGCTAACGC", "GCTATGCGGCTATACGC"),
    ("kühler schrank", "schüler krank"),
    ("the longest", "longest day"),
    ("nicht ausgeloggt", "licht ausgenockt"),
    ("gurken schaben", "schurkengaben")
]
# %%

def hamming(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        return -1
    dist = 0
    for n in range(len(s1)):
        if s1[n] != s2[n]:
            dist += 1
    return dist

# hamming('GCGTATGAGGCTAACGC', 'GCTATGCGGCTATACGC') = 10
# hamming('kühler schrank', 'schüler krank') = 13
# hamming('the longest', 'longest day') = 11
# hamming('nicht ausgeloggt', 'licht ausgenockt') = 4
# hamming('gurken schaben', 'schurkengaben') = 14

# %%
def levenshtein(s1: str, s2: str) -> (int, str):
    transcript = ""
    D = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    D[0, 1:] = range(1, len(s2) + 1)
    D[1:, 0] = range( 1, len(s1) + 1)

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            delta = 1 if s1[i-1] != s2[j-1] else 0
            D[i, j] = min(
                D[i-1, j] + 1,
                D[i, j-1] + 1,
                D[i-1, j-1] + delta
            )

    i = len(s1)
    j = len(s2)
    while i != 0 and j != 0:
        currCost = D[i, j]
        minCost = min(
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
        )

        if currCost == minCost:
            transcript = "m" + transcript
            i = i - 1
            j = j - 1
        elif minCost == D[i-1, j-1]:
            transcript = "s" + transcript
            i = i - 1
            j = j - 1
        elif minCost == D[i-1, j]:
            transcript = "d" + transcript
            i = i - 1
        elif minCost == D[i, j - 1]:
            transcript = "i" + transcript
            j = j - 1
    
    return (D[len(s1), len(s2)], transcript)

# levenshtein('GCGTATGAGGCTAACGC', 'GCTATGCGGCTATACGC') = 3 (mmdmmmmsmmmmmimmmm)
# levenshtein('kühler schrank', 'schüler krank') = 6 (ssmimmmmsddmmmm)
# levenshtein('the longest', 'longest day') = 8 (ddddmmmmmmmiiii)
# levenshtein('nicht ausgeloggt', 'licht ausgenockt') = 4 (smmmmmmmmmmsmssm)
# levenshtein('gurken schaben', 'schurkengaben') = 7 (siimmmmmsdddmmmm)

# %%
# Assignment Pt. 2: Auto-Correct

def suggest(w: str, dist, max_cand=5) -> list:
    """
    w: word in question
    dist: edit distance to use
    max_cand: maximum of number of suggestions

    returns a list of tuples (word, dist, score) sorted by score and distance"""
    path = "../../extern/NLCD/"
    with open(os.path.join(path, 'count_1w.txt')) as f:
        lines = f.read().splitlines()

    names = []
    count = []
    prob = []
    sum = 0
    for line in lines:
        split_line = line.split('\t')
        names.append(split_line[0])
        count.append(int(split_line[1]))
        #Berechnet die Summe aller Wörter
        sum += int(split_line[1])

    for elem in count:
        #Wahrscheinlichkeiten berechnen
        prob.append(elem/sum)

    scores = []
    edit_dist = []
    for i in range(0, len(names)):
        edit_dist_elem, _ = dist(w, names[i])
        edit_dist.append(edit_dist_elem)

        # Scores ist die Gewichtung aus der Bedingten Ws, dass w bestimmtes ist P(w, names[i])
        # und dass das bestimmte Wort allgemein vorkommt P(names[i]). (niedriger Score heißt wahrscheinlicher)
        scores.append((edit_dist_elem / len(w) * (1-prob[i])))

    a = scores[:][0]
    # Scores sortieren (aufsteigend)
    sorted_ind = np.argsort(scores)
    return_arr = []
    for i in range(0, max_cand):
        # Rückgabe des (Wortes, Distanz, Scores)
        return_arr.append((names[sorted_ind[i]], edit_dist[sorted_ind[i]], scores[sorted_ind[i]]))
    return(return_arr)

examples = [
    "pirates",    # in-voc
    "pirutes",    # pirates?
    "continoisly",  # continuosly?
]

# for w in examples:
#     print(w, suggest(w, levenshtein, max_cand=3))

# sample result; your scores may vary!
# pirates [('pirates', 0, -11.408058827802126)]
# pirutes [('pirates', 1, -11.408058827802126), ('minutes', 2, -8.717825438953103), ('viruses', 2, -11.111468702571859)]
# continoisly [('continously', 1, -15.735337826575178), ('continuously', 2, -11.560071979871001), ('continuosly', 2, -17.009283000138204)]

# %%
# Assignment Pt. 3: Needleman-Wunsch

def keyboardsim(s1: str, s2: str) -> float:
    s1 = s1.lower()
    s2 = s2.lower()
    if s1 == s2:
        return 2.0
    res = nw(s1, s2, 0.4, sim)


def sim(c1: chr, c2: chr) -> float:

    keyboard = [
        ['1','2','3','4','5','6','7','8','9','0'],
        ['q','w','e','r','t','y','u','i','o','p'],
        ['a','s','d','f','g','h','j','k','l',';'],
        ['z','x','c','v','b','n','m', ',', '.', '/'],
        [' ']]

    print(c1)
    print(c2)


    c1_pos = [(index, row.index(c1)) for index, row in enumerate(keyboard) if c1 in row]
    c2_pos = [(index, row.index(c2)) for index, row in enumerate(keyboard) if c2 in row]

    print(c1_pos)
    print(c2_pos)

    row_diff = abs(c1_pos[0][0] - c2_pos[0][0])
    col_diff = abs(c1_pos[0][1] - c2_pos[0][1])

    total_diff = 1

    if row_diff == 0:
        if col_diff == 0:
            return total_diff
    elif row_diff == 0:
        total_diff = 1 / col_diff
    
    return total_diff

    # TODO

def nw(s1: str, s2: str, d: float, sim) -> float:
    D = np.zeros((len(s1) + 1, len(s2) + 1), dtype=float)
    D[0, 1:] = range(1, len(s2) + 1); D[0, 1:] *= d
    D[1:, 0] = range(1, len(s1) + 1); D[1:, 0] *= d

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            cs = D[i-1, j-1] + sim(s1[i-1], s2[j-1])
            cd = D[i-1, j] + d
            ci = D[i, j-1] + d
            D[i, j] = max(cs, cd, ci)
    return D[len(s1), len(s2)]

# How does your suggest function behave with nw and a keyboard-aware similarity?

# %%
