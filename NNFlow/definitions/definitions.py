from __future__ import absolute_import, division, print_function


def jet_btag_category():

    conditions_dict = {'j=4b=3'   : '(N_Jets == 4 and N_BTagsM == 3)',
                       'j=4b=4'   : '(N_Jets == 4 and N_BTagsM == 4)',
                       'j=5b=3'   : '(N_Jets == 5 and N_BTagsM == 3)',
                       'j=5b>=4'  : '(N_Jets == 5 and N_BTagsM >= 4)',
                       'j>=6b=2'  : '(N_Jets >= 6 and N_BTagsM == 2)',
                       'j>=6b>=2' : '(N_Jets >= 6 and N_BTagsM >= 2)',
                       'j>=6b=3'  : '(N_Jets >= 6 and N_BTagsM == 3)',
                       'j>=6b>=3' : '(N_Jets >= 6 and N_BTagsM >= 3)',
                       'j>=6b>=4' : '(N_Jets >= 6 and N_BTagsM >= 4)',
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

    weight_list = ['Weight', 'Weight_CSV', 'Weight_PU']

    return weight_list


def generator_level_variables():

    generator_level_variables_list = ['GenHiggs_DecProd1_Eta',
                                      'GenHiggs_DecProd1_PDGID',
                                      'GenHiggs_DecProd1_Pt',
                                      'GenHiggs_DecProd2_Eta',
                                      'GenHiggs_DecProd2_PDGID',
                                      'GenHiggs_DecProd2_Pt',
                                      'GenHiggs_Eta',
                                      'GenHiggs_Phi',
                                      'GenHiggs_Pt',
                                      'Prescale_HLT_Ele27_eta2p1_WPTight_Gsf_vX',
                                      'Prescale_HLT_IsoMu24_vX',
                                      'Prescale_HLT_IsoTkMu24_vX',
                                      'Evt_ID',
                                      'Evt_Odd',
                                      'Evt_Lumi',
                                      'Evt_Run',
                                      'Evt_Phi_GenMET',
                                      'Evt_Pt_GenMET',
                                      'GenEvt_TTxId_FromProducer',
                                      'N_GenTopHad',
                                      'N_GenTopLep',
                                      'N_GoodTags',
                                      'N_GoodTagsM',
                                      'N_MisTags',
                                      'N_MisTagsM',
                                      'N_TotalTags',
                                      'GenEvt_I_TTPlusBB',
                                      'GenEvt_I_TTPlusCC',
                                      'TTBB_GenEvt_I_TTPlusBB',
                                      'TTBB_GenEvt_I_TTPlusCC',
                                      'TTBB_GenEvt_TTxId_FromProducer',
                                      'Triggered_HLT_Ele27_eta2p1_WPTight_Gsf_vX',
                                      'Triggered_HLT_IsoMu24_vX',
                                      'Triggered_HLT_IsoTkMu24_vX',
                                      'GenTopHad_Eta',
                                      'GenTopHad_Phi',
                                      'GenTopHad_Pt',
                                      'GenTopLep_Eta',
                                      'GenTopLep_Phi',
                                      'GenTopLep_Pt',
                                      'Jet_GenJet_Eta',
                                      'Jet_GenJet_Pt',
                                      'Jet_Flav',
                                      'Jet_PartonFlav',
                                      'Jet_PileUpID',
                                      'Reco_highest_TopAndWHadLikelihood',
                                      'Reco_existingAll',
                                      'Reco_existingH',
                                      'Reco_existingN',
                                      'Reco_existingW',
                                      'Reco_foundAll_with_TTBBLikelihood',
                                      'Reco_foundAll_with_TTBBLikelihoodTimesME',
                                      'Reco_foundAll_with_TTHLikelihood',
                                      'Reco_foundAll_with_TTHLikelihoodTimesME',
                                      'Reco_foundAll_with_TTLikelihood',
                                      'Reco_foundAll_with_TTLikelihood_comb',
                                      'Reco_foundH_with_TTBBLikelihood',
                                      'Reco_foundH_with_TTBBLikelihoodTimesME',
                                      'Reco_foundH_with_TTHLikelihood',
                                      'Reco_foundH_with_TTHLikelihoodTimesME',
                                      'Reco_foundH_with_TTLikelihood',
                                      'Reco_foundH_with_TTLikelihood_comb',
                                      'Reco_foundN_with_TTBBLikelihood',
                                      'Reco_foundN_with_TTBBLikelihoodTimesME',
                                      'Reco_foundN_with_TTHLikelihood',
                                      'Reco_foundN_with_TTHLikelihoodTimesME',
                                      'Reco_foundN_with_TTLikelihood',
                                      'Reco_foundN_with_TTLikelihood_comb',
                                      'Reco_foundW_with_TTBBLikelihood',
                                      'Reco_foundW_with_TTBBLikelihoodTimesME',
                                      'Reco_foundW_with_TTHLikelihood',
                                      'Reco_foundW_with_TTHLikelihoodTimesME',
                                      'Reco_foundW_with_TTLikelihood',
                                      'Reco_foundW_with_TTLikelihood_comb',
                                      'Reco_highest_TTHLikelihoodTimesME',
                                      'Reco_highest_TTLikelihood',
                                      'Reco_highest_TTHLikelihood',
                                      'Reco_highest_TTBBLikelihood',
                                      'Reco_highest_TTBBLikelihoodTimesME',
                                      'Reco_highest_TTLikelihood_comb',
                                      'Gen_highest_TopAndWHadLikelihood',
                                      'Gen_existingAll',
                                      'Gen_existingH',
                                      'Gen_existingN',
                                      'Gen_existingW',
                                      'Gen_foundAll_with_TTBBLikelihood',
                                      'Gen_foundAll_with_TTBBLikelihoodTimesME',
                                      'Gen_foundAll_with_TTHLikelihood',
                                      'Gen_foundAll_with_TTHLikelihoodTimesME',
                                      'Gen_foundAll_with_TTLikelihood',
                                      'Gen_foundAll_with_TTLikelihood_comb',
                                      'Gen_foundH_with_TTBBLikelihood',
                                      'Gen_foundH_with_TTBBLikelihoodTimesME',
                                      'Gen_foundH_with_TTHLikelihood',
                                      'Gen_foundH_with_TTHLikelihoodTimesME',
                                      'Gen_foundH_with_TTLikelihood',
                                      'Gen_foundH_with_TTLikelihood_comb',
                                      'Gen_foundN_with_TTBBLikelihood',
                                      'Gen_foundN_with_TTBBLikelihoodTimesME',
                                      'Gen_foundN_with_TTHLikelihood',
                                      'Gen_foundN_with_TTHLikelihoodTimesME',
                                      'Gen_foundN_with_TTLikelihood',
                                      'Gen_foundN_with_TTLikelihood_comb',
                                      'Gen_foundW_with_TTBBLikelihood',
                                      'Gen_foundW_with_TTBBLikelihoodTimesME',
                                      'Gen_foundW_with_TTHLikelihood',
                                      'Gen_foundW_with_TTHLikelihoodTimesME',
                                      'Gen_foundW_with_TTLikelihood',
                                      'Gen_foundW_with_TTLikelihood_comb',
                                      'Gen_highest_TTBBLikelihood',
                                      'Gen_highest_TTBBLikelihoodTimesME',
                                      'Gen_highest_TTHLikelihood',
                                      'Gen_highest_TTHLikelihoodTimesME',
                                      'Gen_highest_TTLikelihood',
                                      'Gen_highest_TTLikelihood_comb']


    return generator_level_variables_list
