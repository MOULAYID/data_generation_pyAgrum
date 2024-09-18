import pyAgrum as gum
import random
import numpy as np
# -------------------------------------------------------------------------------------
def mar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_1'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3,4,5,6,7,8,9,10]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100]))
    vC=bn.add(gum.IntegerVariable('C','C',[4,7]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    # bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.05,0.05])
    bn.cpt(vB)[:]=[ 0.8, 0.2]
    bn.cpt(vC)[:]=[ 0.1, 0.9]
    bn.cpt(vIB)[:]=[ [1,0],[o, r],[o, r],[o, r],[1,0],[o, r],[o, r],[o, r],[1,0],[o, r]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title

# -------------------------------------------------------------------------------------
def mar_2(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_2'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3,4,5,6,7,8,9,10]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000]))
    vC=bn.add(gum.IntegerVariable('C','C',[4,7, 9]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vC,vB)
    # bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.05,0.05])
    # bn.cpt(vB)[:]=[ 0.8, 0.2]
    bn.cpt(vC)[:]= random_triplet_sum_to_one()
    # [ 0.235489, 0.474295, 0.290216]

    for i in range(3):
      for j in range(10):
        if i + 1 < j :
          bn.cpt(vB)[i, j, :] = random_triplet_sum_to_one()
          # [0.33, 0.33, 0.33]
        else:
          bn.cpt(vB)[i, j, :] = random_triplet_sum_to_one()

    # bn.cpt(vIB)[:]=[ [1,0],[o, r],[o, r],[o, r],[1,0],[o, r],[o, r],[o, r],[1,0],[o, r]]
    bn.cpt(vIB)[:]=[0.2,0.8]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_3(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_3'

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    alpha1 = 0.4
    alpha2 = 0.2
    alpha3 = 0.4
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.5, 0, 0],[0, 0.3, 0.3, 0.4, 0], [0.01, 0.19, 0, 0, 0.8]]

    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]

    var_indicator = {'B':'IB'}


    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_4(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_4'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.4,0.2, 0.4])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.5, 0, 0],[0, 0.3, 0.3, 0.4, 0], [0, 0, 0, 0, 1]]
    bn.cpt(vIB)[:]=[ [o, r],[1, 0], [o, r]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_5(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_5'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[13,55, 100, 999]))
    vC=bn.add(gum.IntegerVariable('C','C',[3,5, 100]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vC,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.2, 0.3])
    bn.cpt(vC).fillWith([0.3, 0.2, 0.5])
    bn.cpt(vB)[:]=[ [0, 0, 1, 0],[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    bn.cpt(vIB)[:]=[ [o, r],[1, 0], [o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_6(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_6'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[13,55, 100, 999]))
    vC=bn.add(gum.IntegerVariable('C','C',[3,5, 100]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vC)
    bn.addArc(vC,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.5,0.2, 0.3])
    # bn.cpt(vC).fillWith([0.3, 0.2, 0.5])
    bn.cpt(vB)[:]=[ [0, 0, 1, 0],[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    bn.cpt(vC)[:]=[ [0, 0, 1],[0.25, 0.5, 0.25], [0.5, 0.25, 0.25]]
    bn.cpt(vIB)[:]=[ [o, r],[o, r], [o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_7(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_7'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000, 10000, 100000]))
    vC=bn.add(gum.IntegerVariable('C','C',[11,18,33]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    # bn.addArc(vA,vC)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.4,0.2, 0.4])
    bn.cpt(vB)[:]=[ [0.2, 0.2, 0, 0, 0.6],[0, 0.3, 0.3, 0.4, 0], [0.4, 0.1, 0.2, 0.2, 0.1]]
    bn.cpt(vC)[:]= [0.25, 0.25, 0.5,]
    bn.cpt(vIB)[:]=[ [o, r],[1, 0], [o, r]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_8(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_8'

    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000, 10000, 100000]))
    vC=bn.add(gum.IntegerVariable('C','C',[11,18,33]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.4,0.2, 0.4])
    bn.cpt(vB)[:]=[ [1, 0, 0, 0, 0],[0, 0.3, 0.3, 0.4, 0], [0, 0, 0, 0, 1]]
    bn.cpt(vC).fillWith([0.4,0.3,0.3])
    bn.cpt(vIB)[:]=[ [o, r],[1, 0], [o, r]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_9(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_9'
    o=1-r

    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3,4,5,6]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100, 1000, 10000, 100000, 1000000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.2,0.1, 0.2,0.1,0.1,0.3])
    bn.cpt(vB)[:]=[ [0.3, 0, 0.3, 0.1, 0.1, 0.2],[0, 0.3, 0.3, 0.4, 0,0], [0.2, 0.2, 0.2, 0, 0.2, 0.2], [0.3, 0, 0.3, 0.1, 0.1, 0.2], [0, 0.3, 0.3, 0.4, 0, 0],[0, 0, 0, 0, 0.5, 0.5]]
    bn.cpt(vIB)[:]=[ [0.8, 0.2],[1, 0], [o, r],[o, r], [0.9,0.1],[o, r]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------
def mar_10(r):
    example_title = 'mar_10'
    bn=gum.BayesNet(example_title)

    o=1-r

    vA=bn.add(gum.IntegerVariable('A','A',[2, 13, 18, 20]))
    vB=bn.add(gum.IntegerVariable('B','B',[3, 14, 24, 98]))
    vC=bn.add(gum.IntegerVariable('C','C',[4, 15, 78, 123, 245]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))

    bn.addArc(vA,vC)
    bn.addArc(vB,vC)
    bn.addArc(vB,vIC)
    bn.addArc(vA,vIC)

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

# _____________________________________________________
    bn.cpt(vIC)[0, 0, :] = [1,0]
    bn.cpt(vIC)[0, 1, :] = [1,0]
    bn.cpt(vIC)[0, 2, :] = [o,r]
    bn.cpt(vIC)[0, 3, :] = [o,r]

    bn.cpt(vIC)[1, 0, :] = [1,0]
    bn.cpt(vIC)[1, 1, :] = [1,0]
    bn.cpt(vIC)[1, 2, :] = [o,r]
    bn.cpt(vIC)[1, 3, :] = [o,r]

    bn.cpt(vIC)[2, 0, :] = [1,0]
    bn.cpt(vIC)[2, 1, :] = [1,0]
    bn.cpt(vIC)[2, 2, :] = [o,r]
    bn.cpt(vIC)[2, 3, :] = [o,r]

    bn.cpt(vIC)[3, 0, :] = [1,0]
    bn.cpt(vIC)[3, 1, :] = [1,0]
    bn.cpt(vIC)[3, 2, :] = [o,r]
    bn.cpt(vIC)[3, 3, :] = [o,r]

    # bn.cpt(vIC)[:]=[ [1, 0],[1, 0],[o, r],[o, r]]

    # this dictionary hold all the variables and their indicators
    var_indicator = {'C':'IC'}

    return bn, var_indicator, example_title
    # -------------------------------------------------------------------------------------
def mar_11(r):
    bn=gum.BayesNet('experiment')
    example_title = 'mar_11'

    o=1-r
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,]))
    vB=bn.add(gum.IntegerVariable('B','B',[10,100]))
    vC=bn.add(gum.IntegerVariable('C','C',[4,7]))
    vIC=bn.add(gum.LabelizedVariable('IC','IC',2))

    # defining dependencies between the variables
    # bn.addArc(vA,vB)
    bn.addArc(vA,vB)
    bn.addArc(vA,vIC)
    bn.addArc(vB,vC)
    bn.addArc(vB,vIC)

    # defining the absolute and the conditional probs
    bn.cpt(vA).fillWith([0.2,0.8])
    bn.cpt(vB)[:]=[[0.2,0.8],[0.9,0.1]]
    bn.cpt(vC)[:]=[[0.8, 0.2],[0.8, 0.2]]
    
    bn.cpt(vIC)[0, 0, :] = [0.1, 0.9]
    bn.cpt(vIC)[0, 1, :] = [1, 0]
    # bn.cpt(vIC)[0, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]

    bn.cpt(vIC)[1, 0, :] = [1, 0]
    bn.cpt(vIC)[1, 1, :] = [r, o]
    # bn.cpt(vIC)[1, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]

    # bn.cpt(vIC)[2, 0, :] = [0.1, 0.2, 0.3, 0.1, 0.3]
    # bn.cpt(vIC)[2, 1, :] = [0.1, 0.2, 0.3, 0.3, 0.1]
    # bn.cpt(vIC)[2, 2, :] = [0.1, 0.2, 0.3, 0.1, 0.3]

    var_indicator = {'C':'IC'}

    return bn, var_indicator, example_title
# -------------------------------------------------------------------------------------


def random_triplet_sum_to_one():
    """
    Generate a triplet of positive numbers that sum up to one.

    Returns:
        tuple: A tuple containing three positive numbers (a, b, c) such that a + b + c = 1.
    """
    while True:
        # Generate two random numbers between 0 and 1
        x1 = np.random.rand()
        x2 = np.random.rand()
        
        # Normalize them to make sure their sum is less than 1
        sum_x = x1 + x2
        if sum_x <= 0:
            continue
        
        # Calculate the triplet values
        a = x1 / sum_x
        b = x2 / sum_x
        c = 1 - a - b
        
        # Check that all values are positive
        if c >= 0:
            return (a, b, c)