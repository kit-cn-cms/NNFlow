from __future__ import absolute_import, division, print_function


def jet_btag_category():

    conditions_dict = {'43' : '(N_Jets == 4 and N_BTagsM == 3)',
                       '44' : '(N_Jets == 4 and N_BTagsM == 4)',
                       '53' : '(N_Jets == 5 and N_BTagsM == 3)',
                       '54' : '(N_Jets == 5 and N_BTagsM >= 4)',
                       '62' : '(N_Jets >= 6 and N_BTagsM == 2)',
                       '63' : '(N_Jets >= 6 and N_BTagsM == 3)',
                       '63+': '(N_Jets >= 6 and N_BTagsM >= 3)',
                       '64' : '(N_Jets >= 6 and N_BTagsM >= 4)',
                       }

    variable_list = ['N_Jets', 'N_BTagsM']

    jet_btag_category_dict = {'variables':variable_list, 'conditions':conditions_dict}

    return jet_btag_category_dict


def train_test_data_set():

    conditions_dict = {'train':'Evt_Odd == 1',
                       'test' :'Evt_Odd == 0'}

    variable_list = ['Evt_Odd']

    train_test_data_set_dict = {'variables':variable_list, 'conditions':conditions_dict}

    return train_test_data_set_dict


def ttbar_processes():

    conditions_dict = {'ttbb'   :'GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0',
                       'tt2b'   :'GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0',
                       'ttb'    :'GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0',
                       'ttcc'   :'GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1',
                       'ttlight':'GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0'
                        }

    variable_list = ['GenEvt_I_TTPlusBB', 'GenEvt_I_TTPlusCC']

    ttbar_processes_dict = {'variables':variable_list, 'conditions':conditions_dict}

    return ttbar_processes_dict


def ttbb_processes():

    conditions_dict = {'ttbb'   :'GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0',
                       'tt2b'   :'GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0',
                       'ttb'    :'GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0',
                        }

    variable_list = ['GenEvt_I_TTPlusBB', 'GenEvt_I_TTPlusCC']

    ttbb_processes_dict = {'variables':variable_list, 'conditions':conditions_dict}

    return ttbb_processes_dict


def default_weight_list():

    weight_list = ['Weight_CSV', 'Weight_PU']

    return weight_list
