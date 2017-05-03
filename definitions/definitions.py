from __future__ import absolute_import, division, print_function


def jet_btag_category(category):

    category_dict = {'43' : 'N_Jets == 4 and N_BTagsM == 3',
                     '44' : 'N_Jets == 4 and N_BTagsM == 4',
                     '53' : 'N_Jets == 5 and N_BTagsM == 3',
                     '54' : 'N_Jets == 5 and N_BTagsM >= 4',
                     '62' : 'N_Jets >= 6 and N_BTagsM == 2',
                     '63' : 'N_Jets >= 6 and N_BTagsM == 3',
                     '63+': 'N_Jets >= 6 and N_BTagsM >= 3',
                     '64' : 'N_Jets >= 6 and N_BTagsM >= 4',
                     }

    return category_dict[category]


def ttbar_processes():

    conditions_dict = {'ttbb'   :'GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0',
                       'tt2b'   :'GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0',
                       'ttb'    :'GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0',
                       'ttcc'   :'GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1',
                       'ttlight':'GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0'
                        }

    return conditions_dict
