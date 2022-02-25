"""
Created on 10:22 at 02/06/2021
@author: bo
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_reference_data(path="../bacteria/", show=False, tds_dir=None, save=False):
    use_string = ["reference", "finetune", "test"]
    data_group = {}
    label_group = {}
    for single_string in use_string:
        data_group[single_string] = np.load(path + "X_%s.npy" % single_string, allow_pickle=True)
        label_group[single_string] = np.load(path + "y_%s.npy" % single_string).astype(np.int32)
    label_name = get_label_string()
    if show:
        wavenum = np.load(path + "/wavenumbers.npy")
        label_unique = np.unique(label_group["reference"])
        select = np.random.choice(label_unique, 10, replace=False).astype(np.int32)
        for s in select:
            color_group = ["r", "g", "b"]
            with sns.axes_style("darkgrid"):
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                for i, s_c in enumerate(color_group):
                    ax.plot([], [], s_c, label=use_string[i])
                ax.legend(loc='best', fontsize=8)
                j = 0
                for s_iter, s_string in enumerate(use_string):
                    _index = np.where(label_group[s_string] == s)[0]
                    for s_tr in np.random.choice(_index, 3, replace=False):
                        ax.plot(wavenum, data_group[s_string][s_tr] + j, color=color_group[s_iter])
                        j += 1
                ax.set_title("Class %s (%d)" % (label_name[s], s), fontsize=8)
                if save:
                    plt.savefig(tds_dir + "/%s.pdf" % label_name[s],
                                pad_inches=0, bbox_inches='tight')
    return [data_group["reference"], label_group["reference"]], \
            [data_group["finetune"], label_group["finetune"]], \
            [data_group["test"], label_group["test"]], label_name


def get_label_string():
    ORDER = [16, 17, 14, 18, 15, 20, 21, 24, 23, 26, 27, 28, 29, 25, 6, 7, 5, 3, 4,
             9, 10, 2, 8, 11, 22, 19, 12, 13, 0, 1]

    STRAINS = {}
    STRAINS[0] = "C. albicans"
    STRAINS[1] = "C. glabrata"
    STRAINS[2] = "K. aerogenes"
    STRAINS[3] = "E. coli 1"
    STRAINS[4] = "E. coli 2"
    STRAINS[5] = "E. faecium"
    STRAINS[6] = "E. faecalis 1"
    STRAINS[7] = "E. faecalis 2"
    STRAINS[8] = "E. cloacae"
    STRAINS[9] = "K. pneumoniae 1"
    STRAINS[10] = "K. pneumoniae 2"
    STRAINS[11] = "P. mirabilis"
    STRAINS[12] = "P. aeruginosa 1"
    STRAINS[13] = "P. aeruginosa 2"
    STRAINS[14] = "MSSA 1"
    STRAINS[15] = "MSSA 3"
    STRAINS[16] = "MRSA 1 (isogenic)"
    STRAINS[17] = "MRSA 2"
    STRAINS[18] = "MSSA 2"
    STRAINS[19] = "S. enterica"
    STRAINS[20] = "S. epidermidis"
    STRAINS[21] = "S. lugdunensis"
    STRAINS[22] = "S. marcescens"
    STRAINS[23] = "S. pneumoniae 2"
    STRAINS[24] = "S. pneumoniae 1"
    STRAINS[25] = "S. sanguinis"
    STRAINS[26] = "Group A Strep."
    STRAINS[27] = "Group B Strep."
    STRAINS[28] = "Group C Strep."
    STRAINS[29] = "Group G Strep."

    return [STRAINS[i] for i in ORDER]


def get_antibiotics_string(check=False):
    ATCC_GROUPINGS = {3: 0,
                      4: 0,
                      9: 0,
                      10: 0,
                      2: 0,
                      8: 0,
                      11: 0,
                      22: 0,
                      12: 2,
                      13: 2,
                      14: 3,  # MSSA
                      18: 3,  # MSSA
                      15: 3,  # MSSA
                      20: 3,
                      21: 3,
                      16: 3,  # isogenic MRSA
                      17: 3,  # MRSA
                      23: 4,
                      24: 4,
                      26: 5,
                      27: 5,
                      28: 5,
                      29: 5,
                      25: 5,
                      6: 5,
                      7: 5,
                      5: 6,
                      19: 1,
                      0: 7,
                      1: 7}

    ab_order = [3, 4, 5, 6, 0, 1, 2, 7]

    antibiotics = {}
    antibiotics[0] = "Meropenem"  # E. coli
    antibiotics[1] = "Ciprofloxacin"  # Salmonella
    antibiotics[2] = "TZP"  # PSA
    antibiotics[3] = "Vancomycin"  # Staph
    antibiotics[4] = "Ceftriaxone"  # Strep pneumo
    antibiotics[5] = "Penicillin"  # Strep + E. faecalis
    antibiotics[6] = "Daptomycin"  # E. faecium
    antibiotics[7] = "Caspofungin"  # Candidas

    if not check:
        return [antibiotics[i] for i in ab_order]
    else:
        return [antibiotics[i] for i in ab_order], ATCC_GROUPINGS

def get_antibiotic_group_simple():
    antibiotic_group = np.array(['Vancomycin', 'Vancomycin', 'Vancomycin', 'Vancomycin',
       'Vancomycin', 'Vancomycin', 'Vancomycin', 'Vancomycin',
       'Penicillin', 'Penicillin', 'Daptomycin', 'Daptomycin',
       'Daptomycin', 'Daptomycin', 'Daptomycin', 'Daptomycin',
       'Daptomycin', 'Meropenem', 'Meropenem', 'Ciprofloxacin',
       'Ciprofloxacin', 'Ciprofloxacin', 'Ciprofloxacin', 'Ciprofloxacin',
       'Ciprofloxacin', 'Ciprofloxacin', 'TZP', 'Ceftriaxone',
       'Caspofungin', 'Caspofungin'])
    return antibiotic_group






