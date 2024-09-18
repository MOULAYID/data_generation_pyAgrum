import pyAgrum as gum
import random 
# import 
# ----------------------------------------------------------------------------
def mcar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = 'MCAR-1'
    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[11,32,43]))
    vB=bn.add(gum.IntegerVariable('B','B',[88,1000, 44444]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.2, 0.3])
    bn.cpt(vB)[:]=[ [0, 0, 1],[0.1, 0.4, 0.5], [0.25, 0.25, 0.5]]

    bn.cpt(vIB)[:]=[ o, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------
def mcar_2(r):
    example_title = 'MCAR-2'
    bn=gum.BayesNet(example_title)

    o = 1-r
    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vD=bn.add(gum.IntegerVariable('D','D',[10,50,88,150]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    # vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vB,vC)
    bn.addArc(vC,vD)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2],[0.2,0.2,0.3,0.3]]
    bn.cpt(vD)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2],[0.1,0.2,0.4,0.3]]

    # bn.cpt(vA).fillWith([0.4,0.4,0.2])
    # bn.cpt(vB).fillWith([0.2,0.2,0.1,0.5])
    # bn.cpt(vC).fillWith([0.2,0.3,0.2,0.3])
    # bn.cpt(vD).fillWith([0.3,0.2,0.3,0.2])
    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIC)[:] = [ o, r]
    # bn.cpt(vIC)[:] = [ o, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'C':'IC'}


    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------
def mcar_3(r):
    bn=gum.BayesNet('experiment')
    example_title = 'MCAR-3'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.4, 0.2, 0.4])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.5, 0, 0],[0, 0.3, 0.3, 0.4, 0], [0.01, 0.19, 0, 0, 0.8]]

    bn.cpt(vIB)[:]=[o, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}


    return bn, var_indicator, example_title

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------