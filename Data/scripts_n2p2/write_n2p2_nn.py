from sfparamgen import SymFuncParamGenerator

my_generator = SymFuncParamGenerator(elements=['O', 'C', 'Pt'], r_cutoff = 6.)

nrad = 5

with open('input.nn.test','w') as testfile:
	my_generator.symfunc_type = 'radial'
	my_generator.generate_radial_params(rule='imbalzano2018', mode='center', nb_param_pairs=nrad+1)
	my_generator.write_settings_overview(fileobj = testfile)
	my_generator.write_parameter_strings(fileobj = testfile)
	my_generator.generate_radial_params(rule='imbalzano2018', mode='shift', nb_param_pairs=nrad-1)
	my_generator.write_settings_overview(fileobj = testfile)
	my_generator.write_parameter_strings(fileobj = testfile)
	
	my_generator.symfunc_type = 'angular_narrow'
	my_generator.zetas = [1.0, 4.0, 16.0]
	my_generator.generate_radial_params(rule='imbalzano2018', mode='center', nb_param_pairs=nrad+1)
	my_generator.write_settings_overview(fileobj = testfile)
	my_generator.write_parameter_strings(fileobj = testfile)

    
