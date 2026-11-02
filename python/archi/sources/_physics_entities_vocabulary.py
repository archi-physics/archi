"""The CMS physics entity vocabulary: techniques, concepts, topics.

180 entities that documents are linked to, so that two documents naming
the same thing become reachable from each other. It is deliberately NOT
model-extracted: every edge it produces is a verbatim match of a declared
alias in document text, so the evidence for any edge is a span you can
point at.

Three kinds:

``technique``
    Something an analysis DOES -- b-tagging, gen matching, unfolding,
    tag-and-probe. The most valuable kind, because a technique is where
    a TWiki recipe and the notes implementing it meet.
``concept``
    A reconstructed or physical object -- W boson, b jet, MET, B meson.
    Distinct from a physics OBJECT GROUP: a muon has a POG, a Z boson
    does not, but both are things an analysis handles.
``topic``
    A search target or measurement type, finer-grained than a PAG --
    dark matter searches within EXO, rare B decays within BPH.

Each entry is ``(code, label, alias_regexes)``. The regexes are matched
case-insensitively against document text. Every term is a public physics
term. Nothing was pruned for being frequent, because it is often the
UNION of entities that selects a document.
"""
from __future__ import annotations

#: 81 techniques. (code, label, alias regexes).
TECHNIQUES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('b_tagging', 'b tagging', (
        '\\bb[- ]?tagg?(ing|ed)\\b',
        '\\bb[- ]?jet identification\\b',
    )),
    ('c_tagging', 'c tagging', (
        '\\bc[- ]?tagg?(ing|ed)\\b',
    )),
    ('deepjet_deepcsv_taggers', 'DeepJet / DeepCSV taggers', (
        '\\bdeepjet\\b',
        '\\bdeepcsv\\b',
        '\\bdeepflavour\\b',
        '\\bCSVv2\\b',
    )),
    ('particlenet', 'ParticleNet', (
        '\\bparticlenet\\b',
    )),
    ('boosted_jet_tagging', 'boosted jet tagging', (
        '\\btop[- ]?tagg?(ing|er)\\b',
        '\\bW[- ]?tagg?(ing|er)\\b',
        '\\bsubstructure tagg',
    )),
    ('jet_substructure', 'jet substructure', (
        '\\bjet substructure\\b',
        '\\bN[- ]?subjettiness\\b',
        '\\btau21\\b',
        '\\bsoft ?drop\\b',
        '\\bpruned mass\\b',
    )),
    ('tag_and_probe', 'tag and probe', (
        '\\btag[- ]and[- ]probe\\b',
        '\\btag ?& ?probe\\b',
        '\\bT&P\\b',
    )),
    ('gen_matching', 'gen matching', (
        '\\bgen[- ]?match(ing|ed)\\b',
        '\\bgenerator[- ]level match',
    )),
    ('pileup_reweighting', 'pileup reweighting', (
        '\\bpile[- ]?up reweight',
        '\\bPU reweight',
    )),
    ('pileup_jet_id', 'pileup jet ID', (
        '\\bpile[- ]?up jet ?ID\\b',
        '\\bPU ?jet ?ID\\b',
    )),
    ('jet_energy_corrections', 'jet energy corrections', (
        '\\bjet energy (correction|scale)\\b',
        '\\bJEC\\b',
        '\\bJES\\b',
    )),
    ('jet_energy_resolution', 'jet energy resolution', (
        '\\bjet energy resolution\\b',
        '\\bJER\\b',
    )),
    ('met_filters', 'MET filters', (
        '\\bMET filter',
        '\\bnoise filter',
    )),
    ('met_recoil_correction', 'MET recoil correction', (
        '\\brecoil correction',
    )),
    ('scale_factors', 'scale factors', (
        '\\bscale factors?\\b',
    )),
    ('unfolding', 'unfolding', (
        '\\bunfold(ing|ed)\\b',
        '\\bTUnfold\\b',
        "\\bD'?Agostini\\b",
        '\\bresponse matrix\\b',
        '\\bmigration matrix\\b',
    )),
    ('abcd_method', 'ABCD method', (
        '\\bABCD method\\b',
        '\\bABCD\\b',
    )),
    ('matrix_method', 'matrix method', (
        '\\bmatrix method\\b',
    )),
    ('fake_rate_estimation', 'fake rate estimation', (
        '\\bfake rate\\b',
        '\\bmisidentification rate\\b',
    )),
    ('sideband_estimation', 'sideband estimation', (
        '\\bside[- ]?band\\b',
    )),
    ('control_region', 'control region', (
        '\\bcontrol region\\b',
        '\\bcontrol sample\\b',
    )),
    ('template_fit', 'template fit', (
        '\\btemplate fit\\b',
        '\\btemplate morphing\\b',
    )),
    ('data_driven_background', 'data driven background', (
        '\\bdata[- ]driven\\b',
    )),
    ('maximum_likelihood_fit', 'maximum likelihood fit', (
        '\\bmaximum likelihood\\b',
        '\\bML fit\\b',
    )),
    ('profile_likelihood', 'profile likelihood', (
        '\\bprofile likelihood\\b',
        '\\bprofiled? nuisance',
    )),
    ('simultaneous_fit', 'simultaneous fit', (
        '\\bsimultaneous fit\\b',
    )),
    ('cls_limit_setting', 'CLs limit setting', (
        '\\bCL_?s\\b',
        '\\basymptotic (CL|limit)',
        '\\bmodified frequentist\\b',
    )),
    ('combine_tool', 'Combine tool', (
        '\\bcombine (tool|harvester)\\b',
        '\\bHiggsAnalysis/CombinedLimit\\b',
        '\\bdatacard',
    )),
    ('asimov_dataset', 'Asimov dataset', (
        '\\bAsimov\\b',
    )),
    ('look_elsewhere_effect', 'look elsewhere effect', (
        '\\blook[- ]elsewhere\\b',
        '\\bglobal significance\\b',
    )),
    ('goodness_of_fit', 'goodness of fit', (
        '\\bgoodness[- ]of[- ]fit\\b',
        '\\bsaturated model\\b',
    )),
    ('bdt', 'BDT', (
        '\\bBDT\\b',
        '\\bboosted decision tree',
    )),
    ('deep_neural_network', 'deep neural network', (
        '\\bDNN\\b',
        '\\bdeep neural network',
        '\\bneural network',
    )),
    ('mva_discriminant', 'MVA discriminant', (
        '\\bMVA\\b',
        '\\bmultivariate (analysis|discriminant|technique)',
    )),
    ('tmva', 'TMVA', (
        '\\bTMVA\\b',
    )),
    ('kinematic_fit', 'kinematic fit', (
        '\\bkinematic fit',
    )),
    ('invariant_mass_reconstruction', 'invariant mass reconstruction', (
        '\\binvariant mass (reconstruction|spectrum|distribution)\\b',
        '\\breconstructed invariant mass\\b',
    )),
    ('transverse_mass', 'transverse mass', (
        '\\btransverse mass\\b',
        '\\bMT2\\b',
    )),
    ('svfit_tau_mass_reconstruction', 'SVfit / tau mass reconstruction', (
        '\\bSVfit\\b',
        '\\bcollinear approximation\\b',
    )),
    ('lepton_isolation', 'lepton isolation', (
        '\\b(relative |mini)?isolation\\b',
        '\\bPF isolation\\b',
    )),
    ('effective_area_correction', 'effective area correction', (
        '\\beffective area\\b',
        '\\brho correction\\b',
    )),
    ('primary_vertex_reconstruction', 'primary vertex reconstruction', (
        '\\bprimary vert(ex|ices)\\b',
    )),
    ('secondary_vertex', 'secondary vertex', (
        '\\bsecondary vert(ex|ices)\\b',
        '\\bdisplaced vert',
    )),
    ('impact_parameter', 'impact parameter', (
        '\\bimpact parameter\\b',
        '\\bIP significance\\b',
        '\\bdxy\\b',
        '\\bdz\\b',
    )),
    ('particle_flow', 'particle flow', (
        '\\bparticle[- ]flow\\b',
        '\\bPF candidate',
    )),
    ('track_reconstruction', 'track reconstruction', (
        '\\btrack reconstruction\\b',
        '\\btracking efficiency\\b',
    )),
    ('trigger_efficiency', 'trigger efficiency', (
        '\\btrigger efficiency\\b',
    )),
    ('hlt_path', 'HLT path', (
        '\\bHLT[_ ]',
        '\\bhigh[- ]level trigger\\b',
    )),
    ('l1_trigger_seed', 'L1 trigger seed', (
        '\\bL1 (seed|trigger)\\b',
    )),
    ('l1_prefiring', 'L1 prefiring', (
        '\\bprefir(ing|e)\\b',
    )),
    ('luminosity_measurement', 'luminosity measurement', (
        '\\bluminosity (measurement|uncertainty|calibration)\\b',
        '\\bvan der Meer\\b',
    )),
    ('nuisance_parameters', 'nuisance parameters', (
        '\\bnuisance parameter',
    )),
    ('shape_uncertainty', 'shape uncertainty', (
        '\\bshape (uncertaint|systematic)',
    )),
    ('closure_test', 'closure test', (
        '\\bclosure test\\b',
    )),
    ('cut_flow', 'cut flow', (
        '\\bcut[- ]?flow\\b',
    )),
    ('cut_based_selection', 'cut based selection', (
        '\\bcut[- ]based\\b',
    )),
    ('event_categorization', 'event categorization', (
        '\\bcategori[sz]ation\\b',
        '\\bevent categor',
    )),
    ('blinding', 'blinding', (
        '\\bblind(ed|ing)\\b',
        '\\bunblind',
    )),
    ('toy_monte_carlo', 'toy Monte Carlo', (
        '\\btoy (MC|monte carlo|experiment)',
        '\\bpseudo[- ]?experiment',
    )),
    ('bootstrap', 'bootstrap', (
        '\\bbootstrap\\b',
    )),
    ('efficiency_measurement', 'efficiency measurement', (
        '\\befficiency measurement\\b',
        '\\bselection efficiency\\b',
    )),
    ('acceptance', 'acceptance', (
        '\\bacceptance (and efficiency|correction|times efficiency)\\b',
        '\\bgeometric(al)? acceptance\\b',
    )),
    ('tau_embedding', 'tau embedding', (
        '\\bembedd?(ing|ed) (sample|technique|method)',
        '\\btau embedd',
    )),
    ('charge_misidentification', 'charge misidentification', (
        '\\bcharge (mis[- ]?id|flip|misidentification)',
    )),
    ('top_pt_reweighting', 'top pT reweighting', (
        '\\btop[- ]?p_?T reweight',
    )),
    ('parton_shower_matching', 'parton shower matching', (
        '\\bMLM\\b',
        '\\bFxFx\\b',
        '\\bmatching scale\\b',
        '\\bparton shower match',
    )),
    ('pdf_uncertainty', 'PDF uncertainty', (
        '\\bPDF uncertaint',
        '\\bPDF set',
    )),
    ('scale_variations', 'scale variations', (
        '\\brenormali[sz]ation (and |/)?factori[sz]ation\\b',
        '\\bmu_?R\\b',
        '\\bscale variation',
    )),
    ('isr_fsr_uncertainty', 'ISR FSR uncertainty', (
        '\\bISR\\b',
        '\\bFSR\\b',
        '\\binitial[- ]state radiation\\b',
        '\\bfinal[- ]state radiation\\b',
    )),
    ('rochester_correction', 'Rochester correction', (
        '\\brochester\\b',
        '\\bmuon momentum (scale|correction)',
    )),
    ('electron_energy_scale_and_resolution', 'electron energy scale and resolution', (
        '\\benergy scale and resolution\\b',
        '\\bECAL (scale|smearing)',
    )),
    ('b_tagging_scale_factor_reshaping', 'b tagging scale factor reshaping', (
        '\\breshaping\\b',
        '\\bb[- ]?tag(ging)? scale factor',
    )),
    ('overlap_removal', 'overlap removal', (
        '\\boverlap removal\\b',
        '\\bjet[- ]lepton cleaning\\b',
    )),
    ('jet_id', 'jet ID', (
        '\\bjet ?ID\\b',
    )),
    ('muon_id', 'muon ID', (
        '\\bmuon ?ID\\b',
        '\\btight muon\\b',
        '\\bloose muon\\b',
    )),
    ('electron_id', 'electron ID', (
        '\\belectron ?ID\\b',
        '\\bcut[- ]based electron\\b',
    )),
    ('photon_id', 'photon ID', (
        '\\bphoton ?ID\\b',
    )),
    ('tau_id', 'tau ID', (
        '\\btau ?ID\\b',
        '\\banti[- ]?lepton discriminat',
        '\\bDeepTau\\b',
    )),
    ('nanoaod', 'NanoAOD', (
        '\\bnanoaod\\b',
    )),
    ('miniaod', 'MiniAOD', (
        '\\bminiaod\\b',
    )),
    ('crab_grid_submission', 'CRAB grid submission', (
        '\\bCRAB\\b',
    )),
)

#: 58 concepts. (code, label, alias regexes).
CONCEPTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('w_boson', 'W boson', (
        '\\bW boson\\b',
        '\\bW\\^?[+-]?\\s?boson',
    )),
    ('z_boson', 'Z boson', (
        '\\bZ boson\\b',
    )),
    ('higgs_boson', 'Higgs boson', (
        '\\bHiggs boson\\b',
        '\\bH\\(125\\)',
    )),
    ('top_quark', 'top quark', (
        '\\btop quark\\b',
        '\\bt\\\\bar\\{?t\\}?\\b',
    )),
    ('bottom_quark', 'bottom quark', (
        '\\bbottom quark\\b',
        '\\bb quark\\b',
    )),
    ('charm_quark', 'charm quark', (
        '\\bcharm quark\\b',
        '\\bc quark\\b',
    )),
    ('tau_lepton', 'tau lepton', (
        '\\btau lepton\\b',
        '\\bhadronic tau\\b',
        '\\btau_?h\\b',
    )),
    ('muon', 'muon', (
        '\\bmuons?\\b',
    )),
    ('electron', 'electron', (
        '\\belectrons?\\b',
    )),
    ('photon', 'photon', (
        '\\bphotons?\\b',
    )),
    ('neutrino', 'neutrino', (
        '\\bneutrinos?\\b',
    )),
    ('jet', 'jet', (
        '\\bjets?\\b',
    )),
    ('b_jet', 'b jet', (
        '\\bb[- ]?jets?\\b',
    )),
    ('ak4_jet', 'AK4 jet', (
        '\\bAK4\\b',
        '\\banti[- ]?k_?T.{0,12}0\\.4',
    )),
    ('ak8_jet', 'AK8 jet', (
        '\\bAK8\\b',
        '\\bfat ?jet',
    )),
    ('subjet', 'subjet', (
        '\\bsub[- ]?jets?\\b',
    )),
    ('missing_transverse_momentum', 'missing transverse momentum', (
        '\\bmissing transverse (momentum|energy)\\b',
        '\\bMET\\b',
        '\\bE_?T\\^?miss',
    )),
    ('ht', 'HT', (
        '\\bH_?\\{?T\\}?\\b\\s*(>|<|=|>=|cut|selection|distribution|spectrum)',
        '\\bscalar (sum|p_?T sum)\\b',
        '\\bH_?T\\b',
    )),
    ('dilepton', 'dilepton', (
        '\\bdi[- ]?lepton\\b',
    )),
    ('diphoton', 'diphoton', (
        '\\bdi[- ]?photon\\b',
    )),
    ('dijet', 'dijet', (
        '\\bdi[- ]?jet\\b',
    )),
    ('j_psi', 'J/psi', (
        '\\bJ/?\\\\?psi\\b',
        '\\bJ/Psi\\b',
    )),
    ('upsilon', 'Upsilon', (
        '\\bUpsilon\\b',
        '\\\\Upsilon\\b',
    )),
    ('b_meson', 'B meson', (
        '\\bB meson\\b',
        '\\bB\\^?0\\b(?!\\s*(GeV|TeV))',
        '\\bB\\^?\\{?[+-]\\}?\\s*(meson|hadron|candidate|decay)',
        '\\bB_?s\\^?0?\\s*(meson|candidate|decay|\\\\to)',
    )),
    ('lambda_b_baryon', 'Lambda_b baryon', (
        '\\bLambda_?b\\b',
        '\\\\Lambda_b',
    )),
    ('d_meson', 'D meson', (
        '\\bD meson\\b',
        '\\bD\\^?0\\s*(meson|candidate|decay)',
        '\\bD\\^?\\*\\+?\\s*(meson|candidate|decay)',
    )),
    ('k_short', 'K short', (
        '\\bK\\^?0_?S\\b',
        '\\bK short\\b',
        '\\bK_?S\\^?0\\b',
    )),
    ('quarkonium', 'quarkonium', (
        '\\bquarkoni(um|a)\\b',
    )),
    ('prompt_vs_non_prompt', 'prompt vs non prompt', (
        '\\bnon[- ]?prompt\\b',
        '\\bprompt (J/|\\\\?psi|Upsilon|charmonium|bottomonium)',
    )),
    ('diboson', 'diboson', (
        '\\bdi[- ]?boson\\b',
        '\\bWW\\b',
        '\\bWZ\\b',
        '\\bZZ\\b',
    )),
    ('ttbar', 'ttbar', (
        '\\bt\\\\?bar\\{?t\\}?\\b',
        '\\bttbar\\b',
        '\\bt ?\\\\bar\\{t\\}',
    )),
    ('single_top', 'single top', (
        '\\bsingle top\\b',
        '\\btW\\b',
        '\\bt[- ]channel\\b',
    )),
    ('tth', 'ttH', (
        '\\btt ?H\\b',
        '\\bt\\\\bar\\{t\\}H\\b',
    )),
    ('drell_yan', 'Drell-Yan', (
        '\\bDrell[- ]?Yan\\b',
        '\\bDY\\+?jets\\b',
        '\\bZ/\\\\?gamma\\^?\\*',
    )),
    ('qcd_multijet', 'QCD multijet', (
        '\\bQCD multijet\\b',
        '\\bmulti[- ]?jet background\\b',
    )),
    ('w_jets', 'W+jets', (
        '\\bW\\+ ?jets\\b',
    )),
    ('z_jets', 'Z+jets', (
        '\\bZ\\+ ?jets\\b',
    )),
    ('vector_boson_fusion', 'vector boson fusion', (
        '\\bVBF\\b',
        '\\bvector boson fusion\\b',
    )),
    ('gluon_fusion', 'gluon fusion', (
        '\\bggF\\b',
        '\\bgluon fusion\\b',
    )),
    ('boosted_topology', 'boosted topology', (
        '\\bboosted (topology|regime|jet|object|W|Z|top|Higgs)\\b',
    )),
    ('displaced_vertex_signature', 'displaced vertex signature', (
        '\\bdisplaced (vertex|vertices|jet|lepton|muon|track)\\b',
    )),
    ('long_lived_particle', 'long lived particle', (
        '\\blong[- ]?lived\\b',
        '\\bLLP\\b',
    )),
    ('dark_matter_candidate', 'dark matter candidate', (
        '\\bdark matter\\b',
        '\\bmono[- ]?(jet|photon|Z|X)\\b',
    )),
    ('squark', 'squark', (
        '\\bsquarks?\\b',
    )),
    ('gluino', 'gluino', (
        '\\bgluinos?\\b',
    )),
    ('neutralino', 'neutralino', (
        '\\bneutralinos?\\b',
        '\\bLSP\\b',
    )),
    ('chargino', 'chargino', (
        '\\bcharginos?\\b',
    )),
    ('stop_squark', 'stop squark', (
        '\\bstop (quark|squark|pair)\\b',
        '\\b\\\\tilde\\{t\\}',
    )),
    ('leptoquark', 'leptoquark', (
        '\\bleptoquarks?\\b',
    )),
    ('z_prime', 'Z prime', (
        "\\bZ'\\b",
        '\\bZ ?prime\\b',
    )),
    ('w_prime', 'W prime', (
        "\\bW'\\b",
        '\\bW ?prime\\b',
    )),
    ('graviton', 'graviton', (
        '\\bgravitons?\\b',
    )),
    ('vector_like_quark', 'vector like quark', (
        '\\bvector[- ]?like quark\\b',
        '\\bVLQ\\b',
    )),
    ('heavy_neutral_lepton', 'heavy neutral lepton', (
        '\\bheavy neutral lepton\\b',
        '\\bHNL\\b',
        '\\bsterile neutrino\\b',
    )),
    ('dark_photon', 'dark photon', (
        '\\bdark photon\\b',
    )),
    ('axion_like_particle', 'axion like particle', (
        '\\baxion[- ]?like\\b',
        '\\bALP\\b',
    )),
    ('excited_fermion', 'excited fermion', (
        '\\bexcited (quark|lepton|fermion)\\b',
    )),
    ('extra_dimensions', 'extra dimensions', (
        '\\bextra dimension',
    )),
)

#: 41 topics. (code, label, alias regexes).
TOPICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('h_to_gamma_gamma', 'H to gamma gamma', (
        '\\bH ?\\\\?(to|rightarrow|->) ?\\\\?gamma ?\\\\?gamma\\b',
        '\\bdiphoton (channel|decay)',
        '\\bHgg\\b',
    )),
    ('h_to_zz_to_4_leptons', 'H to ZZ to 4 leptons', (
        '\\bH ?\\\\?(to|rightarrow|->) ?ZZ\\b',
        '\\bfour[- ]?lepton\\b',
        '\\b4l channel\\b',
    )),
    ('h_to_ww', 'H to WW', (
        '\\bH ?\\\\?(to|rightarrow|->) ?WW\\b',
        '\\bHWW\\b',
    )),
    ('h_to_bb', 'H to bb', (
        '\\bH ?\\\\?(to|rightarrow|->) ?b ?\\\\?bar\\{?b\\}?\\b',
        '\\bHbb\\b',
    )),
    ('h_to_tau_tau', 'H to tau tau', (
        '\\bH ?\\\\?(to|rightarrow|->) ?\\\\?tau ?\\\\?tau\\b',
    )),
    ('h_to_mu_mu', 'H to mu mu', (
        '\\bH ?\\\\?(to|rightarrow|->) ?\\\\?mu ?\\\\?mu\\b',
    )),
    ('di_higgs_self_coupling', 'di-Higgs / self coupling', (
        '\\bHH\\b',
        '\\bdi[- ]?Higgs\\b',
        '\\bself[- ]coupling\\b',
    )),
    ('invisible_higgs', 'invisible Higgs', (
        '\\binvisible (Higgs|decay|width)\\b',
    )),
    ('higgs_cp_properties', 'Higgs CP properties', (
        '\\bCP (properties|structure|violation).{0,20}Higgs',
        '\\bHiggs.{0,20}CP\\b',
    )),
    ('ttbar_cross_section', 'ttbar cross section', (
        '\\bt\\\\?bar\\{?t\\}? (production )?cross section\\b',
    )),
    ('top_quark_mass_measurement', 'top quark mass measurement', (
        '\\btop[- ](quark[- ])?mass measurement\\b',
        '\\bm_?t measurement\\b',
    )),
    ('top_spin_correlation', 'top spin correlation', (
        '\\bspin correlation\\b',
        '\\btop polari[sz]ation\\b',
    )),
    ('differential_cross_section', 'differential cross section', (
        '\\bdifferential cross[- ]section (measurement|is measured|analysis)\\b',
        '\\bmeasure.{0,20}differential cross[- ]section',
    )),
    ('charge_asymmetry', 'charge asymmetry', (
        '\\bcharge asymmetry\\b',
    )),
    ('fcnc', 'FCNC', (
        '\\bFCNC\\b',
        '\\bflavour changing neutral current\\b',
    )),
    ('w_mass_measurement', 'W mass measurement', (
        '\\bW[- ]boson mass measurement\\b',
        '\\bm_?W measurement\\b',
    )),
    ('weak_mixing_angle', 'weak mixing angle', (
        '\\bweak mixing angle\\b',
        '\\bsin\\^?2.{0,10}theta_?W\\b',
    )),
    ('vector_boson_scattering', 'vector boson scattering', (
        '\\bvector boson scattering\\b',
        '\\bVBS\\b',
    )),
    ('triboson_production', 'triboson production', (
        '\\btri[- ]?boson\\b',
        '\\bWWW production\\b',
        '\\bVVV production\\b',
        '\\bWWZ\\b',
        '\\bWZZ\\b',
        '\\bZZZ\\b',
    )),
    ('underlying_event', 'underlying event', (
        '\\bunderlying event\\b',
    )),
    ('susy_gluino_squark_search', 'SUSY gluino squark search', (
        '\\bgluino.{0,20}search\\b',
        '\\bsquark.{0,20}search\\b',
    )),
    ('electroweakino_search', 'electroweakino search', (
        '\\belectroweakino\\b',
        '\\bchargino[- ]neutralino\\b',
    )),
    ('compressed_spectra', 'compressed spectra', (
        '\\bcompressed (spectra|scenario)\\b',
    )),
    ('r_parity_violating_susy', 'R parity violating SUSY', (
        '\\bR[- ]?parity violat',
    )),
    ('mono_x_dark_matter_search', 'mono-X dark matter search', (
        '\\bmono[- ]?(jet|photon|Z|W|Higgs|top)\\b',
    )),
    ('heavy_resonance_search', 'heavy resonance search', (
        '\\bheavy resonance\\b',
        '\\bresonance search\\b',
    )),
    ('rare_b_decays', 'rare B decays', (
        '\\bB_?s ?\\\\?(to|rightarrow|->) ?\\\\?mu ?\\\\?mu\\b',
        '\\brare (B )?decay',
    )),
    ('lepton_flavour_universality', 'lepton flavour universality', (
        '\\blepton (flavou?r )?universality\\b',
        '\\bR\\(K\\)',
        '\\bR\\(D\\)',
    )),
    ('cp_violation_in_b', 'CP violation in B', (
        '\\bCP violation\\b.{0,40}\\bB\\b',
        '\\bCP[- ]violating phase\\b',
        '\\bmixing[- ]induced CP\\b',
    )),
    ('quarkonium_production', 'quarkonium production', (
        '\\bquarkonium production\\b',
        '\\bJ/?\\\\?psi production\\b',
    )),
    ('quark_gluon_plasma', 'quark gluon plasma', (
        '\\bquark[- ]gluon plasma\\b',
        '\\bQGP\\b',
    )),
    ('jet_quenching', 'jet quenching', (
        '\\bjet quenching\\b',
        '\\benergy loss\\b',
    )),
    ('elliptic_flow', 'elliptic flow', (
        '\\belliptic flow\\b',
        '\\bflow harmonic',
        '\\bv_?\\{?2\\}?\\s*(coefficient|harmonic)',
    )),
    ('quarkonium_suppression', 'quarkonium suppression', (
        '\\bnuclear modification factor\\b',
        '\\bR_?\\{?AA\\}?\\b',
        '\\bsequential (melting|suppression)\\b',
        '\\bquarkonium suppression\\b',
    )),
    ('small_system_collectivity', 'small system collectivity', (
        '\\bsmall system\\b',
        '\\bridge\\b',
        '\\bpPb\\b',
    )),
    ('ultra_peripheral_collisions', 'ultra peripheral collisions', (
        '\\bultra[- ]?peripheral\\b',
        '\\bUPC\\b',
    )),
    ('diffraction', 'diffraction', (
        '\\bdiffract(ion|ive)\\b',
    )),
    ('exclusive_production', 'exclusive production', (
        '\\bexclusive production\\b',
        '\\bcentral exclusive\\b',
    )),
    ('minimum_bias', 'minimum bias', (
        '\\bminimum[- ]bias\\b',
        '\\bminbias\\b',
        '\\bzero[- ]bias\\b',
    )),
    ('boosted_top_search', 'boosted top search', (
        '\\bboosted top\\b',
    )),
    ('dark_sector_hidden_valley', 'dark sector / hidden valley', (
        '\\bhidden valley\\b',
        '\\bdark sector\\b',
        '\\bdark shower\\b',
    )),
)
#: Every entity, as (kind, code, label, aliases).
ALL: tuple[tuple[str, str, str, tuple[str, ...]], ...] = tuple(
    (kind, code, label, aliases)
    for kind, rows in (
        ("technique", TECHNIQUES),
        ("concept", CONCEPTS),
        ("topic", TOPICS),
    )
    for code, label, aliases in rows
)

#: Node subtype per kind. Kept beside the vocabulary so a new kind
#: cannot be added without deciding what subtype it becomes.
SUBTYPE_BY_KIND = {
    "technique": "physics_technique",
    "concept": "physics_concept",
    "topic": "physics_topic",
}


def entity_node_id(kind: str, code: str) -> str:
    return f"{SUBTYPE_BY_KIND[kind]}:{code}"
