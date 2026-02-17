def parameter (name_parameter = "all"):
    parameter_dict = {
        #mode choice
                      'ASC_bike':-2.5,
                      'ASC_car':0,
                      'mu_mode':1,
                      'beta_time':-0.01,
                      'max_iter_mode_choice':5,
        #traffic assignement
                      'ta_due_algorithm':'bfw',
                      'ta_sto_algorithm': 'bfsle',
                      'max_iter_ta':500,
                      'tolerance':1e-4,
                      'max_route':3

    }
    if name_parameter == "all":
        return parameter_dict
    else:
        return parameter_dict[name_parameter]