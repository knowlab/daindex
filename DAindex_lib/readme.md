## Health inequality metric: allocation-deterioration framework (DA Framework)
We define and quantify health inequalities in a generic resource allocation scenario using a so-called allocation-deterioration framework. The basic idea is to define two indices: allocation index and deterioration index. The allocation index is (to be derived) from the AI model of interest. Conceptually, AI models are abstracted as `resource allocators`, such as predicting the probability of Intensive Care Unit admission. Note that the models themselves do not need to be particularly designed to allocate resources, for example, it could be risk prediction of cardiovascular disease (CVD) among people with diabetes. Essentially, a resource allocator is a computational model that takes patient data as input and outputs a (normalised) score between 0 and 1. We call this score the allocation index. Deterioration index is a score between 0 and 1 to measure the deterioration status of patients. It can be derived from an objective measurement for disease prognosis (i.e., *a marker of prognosis* in epidemiology terminology), such as extensively used comorbidity scores or biomarker measurements like those for CVDs.

When we have the two indices, each patient can then be represented as a point in a two-dimensional space of *allocation index*, *deterioration index*. A group of patients is then translated into a set of points in the space, for which a regression model could be fitted to approximate as a curve in the space. The same could be done for another group. *The area between the two curves is then the deterioration difference between their corresponding patient groups, quantifying the inequalities induced by the `allocator`, i.e., the AI model that produces the allocation index*. The curve with the larger area under it represents the patient group which would be unfairly treated if the allocation index was to be used in allocating resources or services: a patient from this group would be deemed healthier than a patient from another group who is equally ill. The rest of this section gives technical details of realising key components of this conceptual framework.

## Reference
[Honghan Wu](https://knowlab.github.io/), Minhong Wang, Aneeta Sylolypavan, Sarah Wild. “Quantifying Health Inequalities Induced by Data and AI Models”. Accepted by IJCAI-ECAI2022 (April 2022). [slides](https://www.ucl.ac.uk/research-it-services/sites/research_it_services/files/quantifying_health_inequalities_induced_by_data_and_ai_models_0.pdf), [recording](https://web.microsoftstream.com/video/568b2e88-5c21-466e-9bbf-63274048161d), [preprint](https://knowlab.github.io/preprints/DA-AUC-IJCAI22.pdf).

## Usage
0. Create sample data for testing
   ```python
   import pandas as pd
   import numpy as np
   n_size = 100
   
   # generate female data
   
   female_mm = [int(m) for m in np.random.normal(3.2, .5, size=n_size)]
   df_female = pd.DataFrame(dict(mm=female_mm,
                                 gender=['f'] * n_size))
   df_female.head()
   
   # generate male data
   male_mm = [int(m) for m in np.random.normal(3, .5, size=n_size)]
   df_male = pd.DataFrame(dict(mm=male_mm,
                               gender=['m'] * n_size))
   df_male.head()
   
   # merge dataframes
   df = pd.concat([df_female, df_male], ignore_index=True)
   df.info()   
   ```
1. Import the DA index `Util` class
    ```python
    from DAindex import Util as qutil
    ```
   
2. Run inequality analysis between female and male. 
   ```python
   qutil.compare_two_groups(df[df.gender=='f'], df[df.gender=='m'], 'mm', 
                            'female', 'male', '#Multimorbidity', 3, is_discrete=True)
   ```
   You will see something similar to.
   ```python
   ({'overall-prob': 0.9999, 'one-step': 0.7199, 'k-step': 0.054609, '|X|': 100},
   {'overall-prob': 0.9999, 'one-step': 0.42, 'k-step': 0.03195, '|X|': 100},
   0.7092018779342724)
   ```
   The result means the inequality of female vs male is `0.709`.