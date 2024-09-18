import pyAgrum as gum
import random
import numpy as np

def gnrt_marginal_prob(n):
    # Generate n random values
    random_values = np.random.rand(n)
    # Normalize the values so they sum to 1
    probabilities = random_values / np.sum(random_values)
    return probabilities.tolist()

# -----------------------------------------------------------------------------
def recoverable_mnar_1(r):
    bn=gum.BayesNet('experiment')

    o = 1-r

    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100,1000]))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    

    bn.addArc(vA,vB)
    bn.addArc(vB,vIA)
    bn.addArc(vA,vIB)


    bn.cpt(vA).fillWith([0.5,0.3,0.2])
    bn.cpt(vB)[:]=[ [0.6,0.2,0.2], [0.2,0.6,0.2], [0.2,0.2,0.6]]
    bn.cpt(vIA)[:]=[[o, r], [1,0], [1,0]]
    bn.cpt(vIB)[:]=[[o,r], [1,0], [o,r]]

    var_indicator = {'B':'IB', 'A':'IA'}


    return bn, var_indicator, "recoverable_mnar_1"
# -------------------------------------------------------------------------------------
def mnar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mnar_1'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    # bn.addArc(vA,vB)
    bn.addArc(vB,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([1,0])
    bn.cpt(vB)[:]=[ 0.6, 0.4]
    bn.cpt(vIB)[:]=[[o,r], [1,0]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_2(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mnar_2'

    o=1-r

    # defininig the variables and their domains
    # vC=bn.add(gum.LabelizedVariable('C','C',['yes']))
    vA=bn.add(gum.IntegerVariable('A','A',[10,21,23]))
    vB=bn.add(gum.IntegerVariable('B','B',[13,55, 100]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vB,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.2, 0.3])

    bn.cpt(vB)[:]=[ [0, 0.2, 0.8],[0.1, 0.4, 0.5], [0.3, 0.2, 0.5]]

    bn.cpt(vIB)[:]=[ [1, 0],[1, 0], [o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_3(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mnar_3'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,12,23,24,35,46]))
    vB=bn.add(gum.IntegerVariable('B','B',[13,56,103,200,350]))
    vC=bn.add(gum.IntegerVariable('C','C',[1,2,3,4]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vC)
    bn.addArc(vC,vB)
    bn.addArc(vB,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.2, 0.2, 0.1, 0.1, 0.2, 0.2])
    bn.cpt(vC)[:]=[ [0.2, 0.1, 0.2, 0.5],[0.3, 0.2, 0.3, 0.2], [0.5, 0.3, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.2, 0.1, 0.5, 0.2], [0.1, 0.1, 0.3, 0.5]]
    bn.cpt(vB)[:]=[ [0.3, 0.2, 0.2, 0.1, 0.2],[0.1, 0.2, 0.1, 0.2, 0.4], [0.5, 0.2, 0.1, 0.1, 0.1] , [0.5, 0.2, 0.1, 0.1, 0.1]]
    bn.cpt(vIB)[:]=[ [o, r],[o, r], [o, r], [o, r], [o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_4(r):
    example_title = 'MNAR-4'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2, 13, 18, 20]))
    vB=bn.add(gum.IntegerVariable('B','B',[3, 14, 24, 98]))
    vC=bn.add(gum.IntegerVariable('C','C',[4, 15, 78, 123, 245]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))

    bn.addArc(vA,vC)
    bn.addArc(vB,vC)
    bn.addArc(vC,vIC)

    bn.cpt(vA).fillWith([0.2,0.2,0.3,0.3])
    bn.cpt(vB).fillWith([0.3,0.2,0.3,0.2])

    bn.cpt(vC)[0, 0, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[0, 1, :] = [0.1, 0.2, 0.3, 0.3, 0.1]
    bn.cpt(vC)[0, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[0, 3, :] = [0.1, 0.2, 0.3, 0.3, 0.1]

    bn.cpt(vC)[1, 0, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 1, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 2, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 3, :] = [0.2, 0.2, 0.2, 0.2, 0.2]

    bn.cpt(vC)[2, 0, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 1, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 2, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 3, :] = [0.2, 0.2, 0.2, 0.2, 0.2]

    bn.cpt(vC)[3, 0, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[3, 1, :] = [0.1, 0.2, 0.3, 0.3, 0.1]
    bn.cpt(vC)[3, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[3, 3, :] = [0.1, 0.2, 0.3, 0.3, 0.1]

    bn.cpt(vIC)[:]=[ [1, 0],[1, 0],[o, r],[o, r],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'C':'IC'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_5(r):
    example_title = 'mnar_5'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,44,56,77]))
    vB=bn.add(gum.IntegerVariable('B','B',[12,99,134,234]))
    vC=bn.add(gum.IntegerVariable('C','C',[43,78,89,123]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vB,vIB)
    bn.addArc(vC,vIB)

    bn.cpt(vA).fillWith([0.3,0.2,0.3,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.4,0.1],[0.1,0.3,0.4,0.2],[0.2,0.2,0.2,0.4],[0.5,0.1,0.2,0.2]]
    bn.cpt(vC)[:]=[ [0.2,0.4,0.2,0.2],[0.2,0.2,0.4,0.2],[0.3,0.1,0.4,0.2],[0.1,0.1,0.3,0.5]]

    print(bn.cpt(vIB).names)

    bn.cpt(vIB)[0,0,:] = [o, r]
    bn.cpt(vIB)[0,1,:] = [o, r]
    bn.cpt(vIB)[0,2,:] = [o, r]
    bn.cpt(vIB)[0,3,:] = [o, r]

    bn.cpt(vIB)[1,0,:] = [o, r]
    bn.cpt(vIB)[1,1,:] = [o, r]
    bn.cpt(vIB)[1,2,:] = [o, r]
    bn.cpt(vIB)[1,3,:] = [o, r]

    bn.cpt(vIB)[2,0,:] = [o, r]
    bn.cpt(vIB)[2,1,:] = [o, r]
    bn.cpt(vIB)[2,2,:] = [o, r]
    bn.cpt(vIB)[2,3,:] = [o, r]

    bn.cpt(vIB)[3,0,:] = [o, r]
    bn.cpt(vIB)[3,1,:] = [o, r]
    bn.cpt(vIB)[3,2,:] = [o, r]
    bn.cpt(vIB)[3,3,:] = [o, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_6(r):
    example_title = 'mnar_6'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[21,53, 254, 589]))
    vB=bn.add(gum.IntegerVariable('B','B',[33,94, 147, 200, 345]))
    vC=bn.add(gum.IntegerVariable('C','C',[41,65,254,789]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vB,vIB)

    bn.cpt(vA).fillWith([0.2, 0.4, 0.2, 0.2])
    bn.cpt(vC)[:]=[ [0.2,0.5,0.2,0.1],[0.3,0.4,0.2,0.1], [0.4,0.3,0.2,0.1], [0.2,0.2,0.3,0.3]]
    bn.cpt(vB)[:]=[ [0.2,0.3,0.3,0.0,0.2],[0.1,0.0,0.4,0.2,0.3], [0.1,0.3,0.3,0.2,0.1], [0.4,0.0,0.2,0.2,0.2]]

    # the order is important here; and it follows which dependency we declared first
    bn.cpt(vIB)[:] = [ [1, 0],[1, 0],[o, r],[o, r],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_7(r):
    example_title = 'mnar_7'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[21,34,56,78]))
    vB=bn.add(gum.IntegerVariable('B','B',[12,45,198,155]))
    vC=bn.add(gum.IntegerVariable('C','C',[45,98,178,356]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vB,vC)
    bn.addArc(vA,vIC)
    bn.addArc(vC,vIB)

    bn.cpt(vA).fillWith([0.3,0.2,0.3,0.2])
    bn.cpt(vB)[:]=[ [0.4,0.1,0.4,0.1],[0.2,0.5,0.2,0.1],[0.3,0.1,0.3,0.3],[0.2,0.0,0.4,0.4]]
    bn.cpt(vC)[:]=[ [0.2,0.2,0.4,0.2],[0.3,0.1,0.4,0.2],[0.3,0.1,0.3,0.3],[0.2,0.1,0.3,0.4]]

    bn.cpt(vIB)[:] = [ [1, 0],[1, 0],[o,r],[1, 0]]
    bn.cpt(vIC)[:] = [ [o, r],[1,0],[o, r],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_8(r):
    example_title = 'mnar_8'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vB,vIC)
    bn.addArc(vC,vIB)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [1, 0],[1,0],[o,r],[o, r]]

    bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC'}

    return bn, var_indicator, example_title
# ____________________________________________________________

def mnar_9(r):
    example_title = 'mnar_9'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    # bn.addArc(vB,vIC)
    bn.addArc(vC,vIB)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [1, 0],[1,0],[o,r],[o, r]]

    # bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    bn.cpt(vIC)[:] = [1-r, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC'}

    return bn, var_indicator, example_title
# ___________________________________________________________________________________
def mnar_10(r):
    example_title = 'mnar_10'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    # bn.addArc(vB,vIC)
    bn.addArc(vC,vIB)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ 0.3,0.2,0.3,0.2]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [1, 0],[1,0],[o,r],[o, r]]

    # bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    bn.cpt(vIC)[:] = [1-r, r]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_11(r):
    example_title = 'mnar_11'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vD=bn.add(gum.IntegerVariable('D','D',[24,75,88,120]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vID=bn.add(gum.LabelizedVariable('ID','ID',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vB,vIB)
    bn.addArc(vC,vIC)
    bn.addArc(vA,vD)
    bn.addArc(vD,vID)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]
    bn.cpt(vD)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [o, r],[1,0],[1,0],[o, r]]

    bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    bn.cpt(vID)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC',
                     'D':'ID'}

    return bn, var_indicator, example_title
# ____________________________________________________________________________

def mnar_12(r):
    example_title = 'manr_12'
    bn=gum.BayesNet(example_title)

    o = 1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vD=bn.add(gum.IntegerVariable('D','D',[10,50,88,150]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vID=bn.add(gum.LabelizedVariable('ID','ID',2))

    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vB,vIB)
    bn.addArc(vC,vIC)
    bn.addArc(vC,vD)
    bn.addArc(vD,vID)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2]]
    bn.cpt(vD)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2],[0.1,0.2,0.4,0.3]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [o, r],[1,0],[1,0],[o, r]]

    bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    bn.cpt(vID)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC',
                     'D':'ID'}


    return bn, var_indicator, example_title
# ____________________________________________________________________________
def mnar_13(r):
    example_title = 'mnar_13'
    bn=gum.BayesNet(example_title)

    o = 1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2,31,45]))
    vB=bn.add(gum.IntegerVariable('B','B',[3,74,145,555]))
    vC=bn.add(gum.IntegerVariable('C','C',[24,75,88,120]))
    vD=bn.add(gum.IntegerVariable('D','D',[10,50,88,150]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vID=bn.add(gum.LabelizedVariable('ID','ID',2))

    bn.addArc(vA,vB)
    bn.addArc(vB,vC)
    bn.addArc(vB,vIB)
    bn.addArc(vC,vD)
    bn.addArc(vD,vID)
    bn.addArc(vD,vIC)

    bn.cpt(vA).fillWith([0.4,0.4,0.2])
    bn.cpt(vB)[:]=[ [0.3,0.2,0.3,0.2],[0.2,0.3,0.2,0.3],[0.4,0.1,0.4,0.1]]
    bn.cpt(vC)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2],[0.2,0.2,0.3,0.3]]
    bn.cpt(vD)[:]=[ [0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],[0.2,0.2,0.4,0.2],[0.1,0.2,0.4,0.3]]

    # the order is important here; and it follows which dependency we declared first
    # for instance we declared B->IB then C->IB
    bn.cpt(vIB)[:] = [ [o, r],[1,0],[1,0],[o, r]]

    bn.cpt(vIC)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    bn.cpt(vID)[:] = [ [1,0],[o, r],[1,0],[o, r]]
    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB',
                     'C':'IC',
                     'D':'ID'}


    return bn, var_indicator, example_title
# _____________________________________________________________________________________________
def mnar_14(r):
    example_title = 'MNAR_14'
    bn=gum.BayesNet(example_title)

    vA=bn.add(gum.IntegerVariable('A','A',[2, 13, 18, 20]))
    vB=bn.add(gum.IntegerVariable('B','B',[3, 14, 24, 98]))
    vC=bn.add(gum.IntegerVariable('C','C',[1, 100]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))

    bn.addArc(vA,vC)
    bn.addArc(vB,vC)
    bn.addArc(vC,vIC)

    bn.cpt(vA).fillWith([0.2,0.2,0.3,0.3])
    bn.cpt(vB).fillWith([0.3,0.2,0.3,0.2])

    for i in range(4):
      for j in range(4):
        if i + 1 < j :
          bn.cpt(vC)[i, j, :] = [0.7, 0.3]
        else:
          bn.cpt(vC)[i, j, :] = [0.6, 0.4]

    r1 = random.random() * r
    r2 = (r-r1*0.624)/0.376

    # bn.cpt(vIC)[:]=[ [1-r ,r], [r, 1-r]]
    bn.cpt(vIC)[:]=[ [1-r1 ,r1], [1-r2, r2]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'C':'IC'}

    return bn, var_indicator, example_title
# ___________________________________________________________________
def mnar_15(r):
    bn=gum.BayesNet('experiment')
    example_title = 'MNAR_15'

    o=1-r

    # defininig the variables and their domains
    # vC=bn.add(gum.LabelizedVariable('C','C',['yes']))
    vA=bn.add(gum.IntegerVariable('A','A',[10,21,23]))
    vB=bn.add(gum.IntegerVariable('B','B',[13,55, 100]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.2, 0.3])

    bn.cpt(vB)[:]=[ [0.8,0.1,0.1],[0.1, 0.8, 0.1], [0.1,0.1,0.8]]

    bn.cpt(vIB)[:]=[ [1, 0],[1, 0], [o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}


    return bn, var_indicator, example_title
# _________________________________________________________
def mnar_16(r):
    example_title = 'mnar_16'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2, 13, 18, 20]))
    vB=bn.add(gum.IntegerVariable('B','B',[3, 14, 24, 98]))
    vC=bn.add(gum.IntegerVariable('C','C',[4, 15, 78, 123, 245]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))

    bn.addArc(vA,vC)
    bn.addArc(vB,vC)
    bn.addArc(vC,vIC)

    bn.cpt(vA).fillWith([0.2,0.2,0.3,0.3])
    bn.cpt(vB).fillWith([0.3,0.2,0.3,0.2])

    bn.cpt(vC)[0, 0, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[0, 1, :] = [0.1, 0.2, 0.3, 0.3, 0.1]
    bn.cpt(vC)[0, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[0, 3, :] = [0.1, 0.2, 0.3, 0.3, 0.1]

    bn.cpt(vC)[1, 0, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 1, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 2, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[1, 3, :] = [0.2, 0.2, 0.2, 0.2, 0.2]

    bn.cpt(vC)[2, 0, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 1, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 2, :] = [0.2, 0.2, 0.2, 0.2, 0.2]
    bn.cpt(vC)[2, 3, :] = [0.2, 0.2, 0.2, 0.2, 0.2]

    bn.cpt(vC)[3, 0, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[3, 1, :] = [0.1, 0.2, 0.3, 0.3, 0.1]
    bn.cpt(vC)[3, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    bn.cpt(vC)[3, 3, :] = [0.1, 0.2, 0.3, 0.3, 0.1]

    bn.cpt(vIC)[:]=[ [1, 0],[1, 0],[o, r],[o, r],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'C':'IC'}

    return bn, var_indicator, example_title