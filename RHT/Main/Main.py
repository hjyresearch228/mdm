import DataSet.HilbertForEdge as model
import DataSet.ml as ml_index
import KDB.kdb_test as kdb
import GridFile.GridFileFor2DSim as gf_sim
import GridFile.GridFileFor2DGeo as gf_geo
import HT.HTFor2DSim as ht_sim
import HT.HTFor2DGeo as ht_geo
import RSMI.RSMIForGeo as rsmi_geo
import RSMI.RSMIForSim as rsmi_sim

if __name__ == '__main__':
    # experiments performed on the synthetic datasets
    model.separate_train()  # training rht-sp and query processing
    model.stack_train()  # training rht-tk and query processing
    kdb.construct_match()  # constructing kdb and query processing
    ml_index.ml_sim()  # training ml and query processing
    gf_sim.Grid_File_Sim()  # constructing Grid and query processing
    ht_sim.HT_Sim()  # constructing HT and query processing
    rsmi_sim.RSMI_Sim()  # training RSMI and query processing
    # experiments performed on the real datasets
    model.separate_train_real()  # training rht-sp and query processing
    model.stack_train_real()  # training rht-tk and query processing
    kdb.construct_match_real()  # constructing kdb and query processing
    ml_index.ml_geo()  # training ml and query processing
    gf_geo.Grid_File_Geo()  # constructing Grid and query processing
    ht_geo.HT_Geo()  # constructing HT and query processing
    rsmi_geo.RSMI_Geo()  # training RSMI and query processing
