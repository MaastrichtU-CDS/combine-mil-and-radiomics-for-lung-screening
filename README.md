Here is code and support materials of paper 'Lung Cancer Screening Using Deep Attention Based Multiple Instance Learning and Radiomics'

 In the Repository of Radiomics features and an Example of TFrecord, you can find the summary of raw radiomics features from LIDC-IDRI dataset (in summary.xlsx file), summary of normalized radiomics (in normalizedsummary.xlsx file) and summary of simulation neagtive bags (in simulationNegativeSummary.xlsx file) and an example of training and testing TFrecord.

 You can use maketfrecord.py to create a new training and testing TFrecord by using normalizedsummary.xlsx and simulationNegativeSummary.xlsx file as mentioned above.

 You can use MILnetwork.py to run and vaildate our proposed methods. 20 times of experiments of using example tfrecord(SA60Train1 and SA60Test1) were executed by the author, AUC results range from 0.72 to 0.88 and mean value is 0.82.
