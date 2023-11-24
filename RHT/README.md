# Data and Source Code Description

## ①Data Sets

There are two data sets, synthetic and real datasets.

1.  The synthetic datasets is placed at SIM directory, which is in text format. This directory contains one phase of data, with a total of 200 trajectories.

2.  The real datasets is placed at GEO directory, which is in text format. This directory contains one phase of data, with a total of 564 trajectories.

## ②How to run the program

The whole program has been completely encapsulated, and we only need to run **Main.py** file to perform all the experiments mentioned in the paper.

Description of the function called in the Main.py file

1. separate_train()：The RHT-SP is constructed on the synthetic data set. We save the indexes and data to DataSet folder.

2.  separate_train_real()：The RHT-SP is constructed on the real data set. We save the indexes and data to DataSet folder.

3.  stack_train()：The RHT-TK is constructed on the synthetic data set. We save the indexes and data to DataSet folder.

4. stack_train_real()：The RHT-TK is constructed on the real data set. We save the indexes and data to DataSet folder.

5. construct_match()：The KDB is constructed on the synthetic data set. We save the indexes and data to KDB folder.

6.  construct_match_real()：The KDB is constructed on the real data set. We save the indexes and data to KDB folder.

7.  ml_sim()：ML runs on the synthetic data set. We save the indexes and data to DataSet folder.

8. Ml_geo()：ML runs on the real data set. We save the indexes and data to DataSet folder.

9. Grid_File_Sim()：Grid runs on the synthetic data set. We save the model to GridFile/SIM folder.

10. Grid_File_Geo()：Grid runs on the real data set. We save the model to GridFile/GEO folder.

11. HT_Sim()：HT runs on the synthetic data set. We save the model to HT/SIM folder.

12. HT_Geo()：HT runs on the real data set. We save the model to HT/GEO folder.

13. RSMI_Sim()：RSMI runs on the synthetic data set. We save the model to RSMI/SIM folder.

14. RSMI_Geo()：RSMI runs on the real data set. We save the model to RSMI/GEO folder.

## ③Introduction of project structure

1. GEO folder：It stores one phase of real data.

2. SIM folder：It stores one phase of synthetic data.

3. Main folder：It stores the Main.py file, which is the entry to the program.

4. DataSet folder：It stores RHT-SP, RHT-TK and ML.

5. KDB folder：It stores the model KDB.

6. GridFile folder：It stores the model Grid.

   - GridFile/GEO/models/EP1：It stores the models trained with one phase of real data.

   - GridFile/SIM/models/EP1：It stores the models trained with one phase of synthetic data.

7. HT folder：It stores the model HT.

   - HT/GEO/ models/EP1：It stores the models trained with one phase of real data.

   - HT/SIM/ models/EP1：It stores the models trained with one phase of synthetic data.

8. RSMI folder: It stores the model RSMI.

   - RSMI/GEO/ models/EP1：It stores the models trained with one phase of real data.

   - RSMI/SIM/ models/EP1：It stores the models trained with one phase of synthetic data.

## ④The correspondence between the experiment stated in the paper and the source code

1.  The section 6.1.1 of the paper corresponds to Line 917 in DataSet/HilbertForEdge.py file.

2.  The section 6.1.2 of the paper corresponds to Line 1258 in DataSet/HilbertForEdge.py file. We adjust the parameter maxError to get the corresponding index size and construction time.

3.  The section 6.1.3 corresponds to Line 1323 and Line 1396 in DataSet/HilbertForEdge.py file. For line 1323, we test the average prediction error of the models on synthetic data, and for line 1396, we test the average prediction error of the models on real data.

4.  For the section 6.2.1 in the paper, we only need to run Main/Main.py file to get the index sizes of different methods.

5.  For the section 6.2.2 in the paper, we only need to run Main/Main.py file to get the index construction time of different methods.

6.  For the section 6.3.1 in the paper, we only need to run Main/Main.py file to get the average number of accessed disk blocks for different methods.

7.  For the section 6.3.2, we only need to run Main/Main.py file to get the recall of the 4 approximate indexes, RHT-SP, RHT-TK, ML and RSMI.

 

 

***If you have any question, please do not hesitate to contact Jingyu Han via jyhan@njupt.edu.cn.\***

​                              

 
