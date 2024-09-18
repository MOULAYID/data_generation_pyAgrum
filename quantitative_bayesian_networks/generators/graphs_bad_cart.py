import pyAgrum as gum
import random
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mnar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_1"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)
    bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    alpha1 = 0.4
    alpha2 = 0.2
    alpha3 = 0.4
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.48, 0, 0.02],[0, 0.3, 0.3, 0.39, 0.01], [0.01, 0.19, 0, 0.02, 0.78]]

    # print(r)
    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]
    bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------------------------------
def bad_cart_mar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_mar_1"
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
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.48, 0, 0.02],[0, 0.3, 0.3, 0.39, 0.01], [0.01, 0.19, 0, 0.02, 0.78]]

    print(r)
    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]

    var_indicator = {'B':'IB'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mnar_2(r):
    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_2"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    alpha1 = 0.4
    alpha2 = 0.2
    alpha3 = 0.4
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.48, 0, 0.02],[0, 0.3, 0.3, 0.39, 0.01], [0.01, 0.19, 0, 0.02, 0.78]]

    # print(r)
    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]

    bn.cpt(vIA).fillWith([1-r1, r1])
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mcar_1(r):
    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mcar_1"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    # vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    # bn.addArc(vA,vIB)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    alpha1 = 0.4
    alpha2 = 0.2
    alpha3 = 0.4
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.48, 0, 0.02],[0, 0.3, 0.3, 0.39, 0.01], [0.01, 0.19, 0, 0.02, 0.78]]

    # print(r)
    # r1 = r/2
    # o1 = 1-r1
    # r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    # bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]

    bn.cpt(vIA).fillWith([1-r, r])
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'A':'IA'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mnar_3(r):
    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_3"
    # defininig the variables and their domains
    # vY=bn.add(gum.IntegerVariable('Y','Y',[4,6,8,10]))
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    # bn.addArc(vY,vA)
    bn.addArc(vA,vB)
    bn.addArc(vB,vIA)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    alpha1 = 0.4
    alpha2 = 0.2
    alpha3 = 0.4
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.2, 0.3, 0.48, 0, 0.02],[0, 0.3, 0.3, 0.39, 0.01], [0.01, 0.19, 0, 0.02, 0.78]]

    # print(r)
    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1],[1 - r2, r2], [o1, r1]]

    bn.cpt(vIB).fillWith([1-r1, r1])
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title

# ----------------------------------------------------------------------------------------------------
def generate_marginal_probabilities(n):
    # Generate n-1 random numbers between 0 and 1
    probs = [random.random() for _ in range(n-1)]
    
    # Sort the probabilities
    probs.sort()
    
    # Calculate the marginal probabilities
    marginal_probs = [probs[0]]
    for i in range(1, n-1):
        marginal_probs.append(probs[i] - probs[i-1])
    marginal_probs.append(1 - probs[-1])
    
    return marginal_probs
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mnar_2_semi_random():
    r =random.uniform(0.1, 0.5)

    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_2_semi_random"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    A_probs_list = generate_marginal_probabilities(3)
    alpha1 = A_probs_list[0]
    alpha2 = A_probs_list[1]
    alpha3 = A_probs_list[2]
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ generate_marginal_probabilities(5),generate_marginal_probabilities(5), generate_marginal_probabilities(5)]

    # print(r)
    r1 = r/2
    o1 = 1-r1
    r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1]]

    bn.cpt(vIA).fillWith([1-r1, r1])
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title
# ----------------------------------------------------------------------------------------------------
def bad_cart_recoverable_mnar_2_total_random():
    r =random.uniform(0.1, 0.5)

    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_2_total_random"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    A_probs_list = generate_marginal_probabilities(3)
    alpha1 = A_probs_list[0]
    alpha2 = A_probs_list[1]
    alpha3 = A_probs_list[2]
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ generate_marginal_probabilities(5),generate_marginal_probabilities(5), generate_marginal_probabilities(5)]

    # print(r)
    # r1 = r/2
    # o1 = 1-r1
    # r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ generate_marginal_probabilities(2), generate_marginal_probabilities(2), generate_marginal_probabilities(2)]

    bn.cpt(vIA).fillWith(generate_marginal_probabilities(2))
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title
# ___________________________________________________________________________________

def bad_cart_recoverable_mnar_2_test():
    # r =random.uniform(0.1, 0.5)

    bn=gum.BayesNet('experiment')
    example_title = "bad_cart_recoverable_mnar_2_test"
    # defininig the variables and their domains
    vA=bn.add(gum.IntegerVariable('A','A',[1,2,3]))
    vB=bn.add(gum.IntegerVariable('B','B',[10, 100, 1000, 10000, 100000]))
    vIB=bn.add(gum.LabelizedVariable('IB','IB',2))
    vIA=bn.add(gum.LabelizedVariable('IA','IA',2))

    # defining dependencies between the variables
    bn.addArc(vA,vB)
    bn.addArc(vA,vIB)
    # bn.addArc(vB,vIA)

    # defining the absolute and the conditional probs
    # A_probs_list = generate_marginal_probabilities(3)
    alpha1 = 0.0999906  
    alpha2 = 0.650279
    alpha3 = 0.24973
    bn.cpt(vA).fillWith([alpha1, alpha2, alpha3])
    bn.cpt(vB)[:]=[ [0.217899, 0.376667, 0.379012, 0.00831871, 0.0181031],[0.110438, 0.0417, 0.0967812, 0.219463, 0.531618], 
    [0.0147481, 0.0511229, 0.019629, 0.718396, 0.196104]]

    # print(r)
    # r1 = r/2
    # o1 = 1-r1
    # r2 = (r - r1 *(alpha1 + alpha3))/alpha2
    bn.cpt(vIB)[:]=[ [0.605381, 0.394619], [0.0165753, 0.983425], [0.928817, 0.071183]]

    bn.cpt(vIA).fillWith([0.606534, 0.393466])
    # bn.cpt(vIA)[:]=[ [o1, r1], [1 - r2, r2], [o1, r1], [1,0], [1,0]]

    var_indicator = {'B':'IB', 'A':'IA'}

    return bn, var_indicator, example_title