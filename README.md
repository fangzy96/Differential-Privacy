# Differential-Privacy
   
Logistic Regression   
1.1 logreg_model.py: this file contains main model and functions of logistic regression   
1.2 logreg_IO.py: run this file in terminal when adding noise to input or outpus in logistic regression   
1.3 logreg_obj.py: run this file in terminal when adding noise to objective function in logistic regression   
1.4 logreg_SGD.py: run this file in terminal when adding noise to SGD in logistic regression   
   
Semi-supervised model   
1.1 semi_main.py: this is the main function of semi-supervised model, run it in terminal when adding noise   
1.2 fnn_model.py: this file contains the fnn model   
1.3 torch_dataset.py: this file contains the dataset of loader   
1.4 teacher.py: this files contains the Teacher class   
1.5 student.py: this files contains the Student class   
   
Other tool file   
1.1 util.py: this files contains functions, for example: read_data, accuracy, split_data and so on   
   
How to run it:   
In order to be convenient for inputing, I use different files for different perturbing, because different perturbing method may need different input parameters   
   
Logistic Regression   
1.1 Objective perturbing:   
Run logreg_obj.py in terminal   
python logreg_obj.py 440data\volcanoes lamda epsilon use_cross_validation   
For example:   
python logreg_obj.py 440data\volcanoes 0.01 0.5   
python logreg_obj.py 440data\volcanoes 0.01 0.5 --no-cv   
   
1.2 Input and Output perturbing:   
Run logreg_IO.py in terminal   
python logreg_IO.py 440data\volcanoes lamda noise_type epsilon learning_rate use_cross_validation   
noise_typy: 0,1,2   
0: without noise   
1: input perturbing   
2: output perturbing   
For example:   
python logreg_IO.py 440data\volcanoes 0.01 1 0.5 0.2   
python logreg_IO.py 440data\volcanoes 0.01 2 0.8 0.2 --no-cv   
   
1.3 SGD perturbing:   
Run logreg_SGD.py in terminal   
python logreg_SGD.py 440data\volcanoes lamda delta epsilon norm_bound learning_rate use_cross_validation   
For example:   
python logreg_SGD.py 440data\volcanoes 0.01 0.1 0.5 1 0.2   
python logreg_SGD.py 440data\volcanoes 0.01 0.1 0.8 2 0.2 --no-cv   
   
Semi-supervised model   
1.1 Counts and input perturbing:   
Run semi_main.py in terminal   
python semi_main.py 440data/volcanoes/volcanoes.data num_teacher epsilon noise_type   
epsilon = 0: no noise   
noise_type: 1,2   
1: Counts perturbing   
2: Input perturbing   
For example:   
python semi_main.py 440data/volcanoes/volcanoes.data 20 0.5 1   
python semi_main.py 440data/volcanoes/volcanoes.data 50 0.5 2   
