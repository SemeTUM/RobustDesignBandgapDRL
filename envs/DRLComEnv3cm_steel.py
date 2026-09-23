import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mph
import gym
import torch
from gym import spaces
import re
import ast
sys.path.append("..")

class VibEnv3D():
    def __init__(self):
        self.action_space = spaces.Box(low=0.1, high=0.99, shape=(10,10))
    def comsolEnv(self,path1):
        client= mph.start(cores=2)
        model= client.create('Model')
        model.clear()
        model=model.java

        model.modelNode().create("comp1") 
        
        model.param().set("kx", "if(k<1,k*pi/L1,if(k<2,pi/L1, (3-k)*pi/L1))");
        model.param().set("ky", "if(k<1,0,if(k<2,(k-1)*pi/L1, (3-k)*pi/L1))");
        model.param().set("k", "0");
        model.param().set("L1", "0.03[m]");
        model.param().set("H", "0.0[m]");
        model.param().set("ra", "0.0[m]");
        
        model.component("comp1").geom().create("geom1", 2);
    
        model.result().table().create("tbl1", "Table");
    
        model.component("comp1").mesh().create("mesh1");
    

        model.component("comp1").geom("geom1").lengthUnit("cm");
        model.component("comp1").geom("geom1").create("r1", "Rectangle");
        model.component("comp1").geom("geom1").feature("r1").set("size", ["3", "3"]);
        model.component("comp1").geom("geom1").feature("r1").set("pos", ["0", "0"]);
        model.component("comp1").geom("geom1").create("r2", "Rectangle");
        model.component("comp1").geom("geom1").feature("r2").set("size", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("r2").set("pos", ["0", "0"]);
        model.component("comp1").geom("geom1").create("arr1", "Array");
        model.component("comp1").geom("geom1").feature("arr1").set("fullsize", ["15", "15"]);
        model.component("comp1").geom("geom1").feature("arr1").set("displ", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("arr1").selection("input").set("r2");
        model.component("comp1").geom("geom1").create("r3", "Rectangle");
        model.component("comp1").geom("geom1").feature("r3").set("size", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("r3").set("pos", ["1.5", "0"]);
        model.component("comp1").geom("geom1").create("arr2", "Array");
        model.component("comp1").geom("geom1").feature("arr2").set("fullsize", ["15", "15"]);
        model.component("comp1").geom("geom1").feature("arr2").set("displ", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("arr2").selection("input").set("r3");
        model.component("comp1").geom("geom1").create("r4", "Rectangle");
        model.component("comp1").geom("geom1").feature("r4").set("size", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("r4").set("pos", ["0", "1.5"]);
        model.component("comp1").geom("geom1").create("arr3", "Array");
        model.component("comp1").geom("geom1").feature("arr3").set("fullsize", ["15", "15"]);
        model.component("comp1").geom("geom1").feature("arr3").set("displ", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("arr3").selection("input").set("r4");
        model.component("comp1").geom("geom1").create("r5", "Rectangle");
        model.component("comp1").geom("geom1").feature("r5").set("size", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("r5").set("pos", ["1.5", "1.5"]);
        model.component("comp1").geom("geom1").create("arr4", "Array");
        model.component("comp1").geom("geom1").feature("arr4").set("fullsize", ["15", "15"]);
        model.component("comp1").geom("geom1").feature("arr4").set("displ", ["0.1", "0.1"]);
        model.component("comp1").geom("geom1").feature("arr4").selection("input").set("r5");
        model.component("comp1").geom("geom1").create("dif1", "Difference");
        model.component("comp1").geom("geom1").feature("dif1").set("intbnd", "off");
        model.component("comp1").geom("geom1").feature("dif1").selection("input").set("arr1", "arr2", "arr3", "arr4", "r1");
        model.component("comp1").geom("geom1").feature("dif1").selection("input2").set(path1);
        model.component("comp1").geom("geom1").run();

        
    
        model.component("comp1").selection().create("box1", "Box");
        model.component("comp1").selection("box1").set("entitydim", "1");
        model.component("comp1").selection().create("box2", "Box");
        model.component("comp1").selection("box2").set("entitydim", "1");
        model.component("comp1").selection().create("box3", "Box");
        model.component("comp1").selection("box3").set("entitydim", "1");
        model.component("comp1").selection().create("box4", "Box");
        model.component("comp1").selection("box4").set("entitydim", "1");
        model.component("comp1").selection().create("box5", "Box");
        model.component("comp1").selection("box5").set("entitydim", "1");
        model.component("comp1").selection().create("box6", "Box");
        model.component("comp1").selection("box6").set("entitydim", "1");
        model.component("comp1").selection("box1").label("x_left");
        model.component("comp1").selection("box1").set("xmin", "-0.01");
        model.component("comp1").selection("box1").set("xmax", "0.01");
        model.component("comp1").selection("box1").set("ymin", "-0.01");
        model.component("comp1").selection("box1").set("ymax", "3.01");
        model.component("comp1").selection("box1").set("condition", "inside");
        model.component("comp1").selection("box2").label("x_right");
        model.component("comp1").selection("box2").set("xmin", "2.99");
        model.component("comp1").selection("box2").set("xmax", "3.01");
        model.component("comp1").selection("box2").set("ymin", "-0.001");
        model.component("comp1").selection("box2").set("ymax", "3.01");
        model.component("comp1").selection("box2").set("condition", "inside");
        model.component("comp1").selection("box3").label("y_up");
        model.component("comp1").selection("box3").set("xmin", "-0.001");
        model.component("comp1").selection("box3").set("xmax", "3.01");
        model.component("comp1").selection("box3").set("ymin", "2.99");
        model.component("comp1").selection("box3").set("ymax", "3.01");
        model.component("comp1").selection("box3").set("condition", "inside");
        model.component("comp1").selection("box4").label("y_down");
        model.component("comp1").selection("box4").set("xmin", "-0.001");
        model.component("comp1").selection("box4").set("xmax", "3.01");
        model.component("comp1").selection("box4").set("ymin", "-0.01");
        model.component("comp1").selection("box4").set("ymax", "0.01");
        model.component("comp1").selection("box4").set("condition", "inside");
        model.component("comp1").selection("box5").label("x_per");
        model.component("comp1").selection("box5").set("inputent", "selections");
        model.component("comp1").selection("box5").set("input", ["box1", "box2"]);
        model.component("comp1").selection("box6").label("y_per");
        model.component("comp1").selection("box6").set("inputent", "selections");
        model.component("comp1").selection("box6").set("input", ["box3", "box4"]);
    
        model.component("comp1").common().create("mpf1", "ParticipationFactors");
    
        model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
        model.component("comp1").physics("solid").create("pc1", "PeriodicCondition", 1);
        model.component("comp1").physics("solid").feature("pc1").selection().named("box5");
        model.component("comp1").physics("solid").create("pc2", "PeriodicCondition", 1);
        model.component("comp1").physics("solid").feature("pc2").selection().named("box6");
    
        model.component("comp1").mesh("mesh1").autoMeshSize(3);
    
        model.result().table("tbl1").comments("Global Evaluation 1");
    
#        model.component("comp1").view("view1").axis().set("xmin", -2.6093924045562744);
#        model.component("comp1").view("view1").axis().set("xmax", 7.472102165222168);
#        model.component("comp1").view("view1").axis().set("ymin", -1.3140366077423096);
#        model.component("comp1").view("view1").axis().set("ymax", 4.63471794128418);
    
        model.component("comp1").physics("solid").feature("lemm1").set("E_mat", "userdef");
        model.component("comp1").physics("solid").feature("lemm1").set("E", "250e9");
        model.component("comp1").physics("solid").feature("lemm1").set("nu_mat", "userdef");
        model.component("comp1").physics("solid").feature("lemm1").set("nu", "0.28");
        model.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef");
        model.component("comp1").physics("solid").feature("lemm1").set("rho", "7850");
        model.component("comp1").physics("solid").feature("pc1").set("PeriodicType", "Floquet");
        model.component("comp1").physics("solid").feature("pc1").set("kFloquet", [["kx"], ["ky"], ["0"]]);
        model.component("comp1").physics("solid").feature("pc2").set("PeriodicType", "Floquet");
        model.component("comp1").physics("solid").feature("pc2").set("kFloquet", [["kx"], ["ky"], ["0"]]);
    
        model.study().create("std1");
        model.study("std1").create("param", "Parametric");
        model.study("std1").create("eig", "Eigenfrequency");
    
        model.sol().create("sol1");
        model.sol("sol1").attach("std1");
        model.sol().create("sol2");
        model.sol("sol2").study("std1");
        model.sol("sol2").label("Parametric Solutions 1");
    
        model.result().numerical().create("gev1", "EvalGlobal");
        model.result().numerical("gev1").set("data", "dset2");
        model.result().create("pg2", "PlotGroup1D");
        model.result("pg2").set("data", "dset2");
        model.result("pg2").create("glob1", "Global");
        model.result("pg2").feature("glob1").set("expr", ["solid.freq"]);
    
        model.study("std1").feature("param").set("pname", ["k"]);
        model.study("std1").feature("param").set("plistarr", ["range(0,0.1,3)"]);
        model.study("std1").feature("param").set("punit", [""]);
        model.study("std1").feature("eig").set("neigs", "20");
        model.study("std1").feature("eig").set("neigsactive", "on");
        model.study("std1").createAutoSequences("jobs");
    
        model.batch("p1").feature("so1").set("psol", "sol2");
    
        model.sol("sol1").createAutoSequence("std1");
    
        model.study("std1").runNoGen();
    
        model.result().numerical("gev1").set("table", "tbl1");
        model.result().numerical("gev1").set("tablecols", "outer");
        model.result().numerical("gev1").set("expr", ["solid.freq"]);
        model.result().numerical("gev1").set("unit", ["Hz"]);
        model.result().numerical("gev1").set("descr", ["Frequency"]);
        model.result().numerical("gev1").set("const", [["solid.refpntx", "0", "Reference point for moment computation, x-coordinate"], ["solid.refpnty", "0", "Reference point for moment computation, y-coordinate"], ["solid.refpntz", "0", "Reference point for moment computation, z-coordinate"]]);
        model.result().numerical("gev1").setResult();
        model.result("pg2").set("ylabel", "Frequency (Hz)");
        model.result("pg2").set("xlog", "on");
        model.result("pg2").set("showlegends", "off");
        model.result("pg2").set("ylabelactive", "off");
        model.result("pg2").feature("glob1").set("unit", ["Hz"]);
        model.result("pg2").feature("glob1").set("descr", ["Frequency"]);
        model.result("pg2").feature("glob1").set("const", [["solid.refpntx", "0", "Reference point for moment computation, x-coordinate"], ["solid.refpnty", "0", "Reference point for moment computation, y-coordinate"], ["solid.refpntz", "0", "Reference point for moment computation, z-coordinate"]]);
        model.result("pg2").feature("glob1").set("xdatasolnumtype", "outer");
        model.result("pg2").feature("glob1").set("xdata", "expr");
        model.result("pg2").feature("glob1").set("xdataexpr", "k");
        model.result("pg2").feature("glob1").set("linewidth", "preference");


        table_str=model.result().table('tbl1').getTableData(1);
        table_str= np.array(table_str, dtype=object)
        #model.save('Model3cm_vibacou')
        client.remove('Model')
        return table_str


    def getValue(self,bndgp):
        #DOI: 10.1098/rsta.2003.1177  (eqn. 3.4)
        bandgap= [str(i) for i in bndgp]
        bandgap = [self.clean_entry(x) for x in bandgap]
        bng = np.array(bandgap).T #(32,20) shape
   
        Obj1=[]

        for i in range(19):
           # J = 2*((min(bng[i+1])**2 - max(bng[i])**2)/(min(bng[i+1])**2 + max(bng[i])**2))
            J = (min(bng[:,i+1]) - max(bng[:,i]))/1000
            local_reward = 10*J if J >0 else J
            Obj1.append(local_reward)
        reward = np.sum(np.real(Obj1))/100
        reward= np.clip(reward, -10, 100)
        print("comsolReward", reward)
        return bng, reward

    def clean_entry(self, entry):
        entry = re.sub(r"[\[\]\n']", " ", entry)
        tokens = entry.split()
        return [self.parse_complex(tok) for tok in tokens]

    def parse_complex(self,s):
        s = s.strip().replace('i', 'j')  # Python uses 'j' for imaginary unit
        try:
            return complex(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return 0.0
