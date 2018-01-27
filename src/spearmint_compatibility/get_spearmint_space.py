import space_manager
import hyperopt
from hyperopt.unif_hparam_sample import Unif_Sampler
from collections import OrderedDict
import copy

def main(num_models, model_types, search_space):
    space = space_manager.get_space(num_models, model_types, search_space)
    us = Unif_Sampler(hyperopt.pyll.as_apply(space))
    
    transformer = hyperopt.base.Bandit(space, do_checks=False)

    


    variables = make_variables(us)

    output_vector = make_output_vect(us)
    
    return variables, output_vector, transformer


def make_variables(us):
    variables = []
    for i in range(len(us.index_names)):
        assert isinstance(us.dists[i], hyperopt.hparam_distribution_sampler.QUniform) or isinstance(us.dists[i], hyperopt.hparam_distribution_sampler.Uniform)
        cur_name = us.index_names[i]
        if isinstance(us.dists[i], hyperopt.hparam_distribution_sampler.QUniform):
            cur_type = "int"
        elif isinstance(us.dists[i], hyperopt.hparam_distribution_sampler.Uniform):
            cur_type = "float"
        cur_min = us.dists[i].a
        cur_max = us.dists[i].b
        variables.append(OrderedDict([('name',cur_name), ('type', cur_type), ('min', cur_min), ('max', cur_max), ('size', 1)]))
        
    for v in variables:
        print v
    print("")
    #sys.exit()
    return variables


def make_output_vect(us):
    #one_samp = us.draw_unif_samp()
    output_vect = copy.deepcopy(us.hparam_out)
    
    for var in us.index_names:
        output_vect[var] = []

    return output_vect
