import pyAgrum as gum
import random
# ---------------------------------------------------------------------------------------------------------
def mar_random_bn_generator(fast_bn,miss_rate,target_variable):
    bn=gum.fastBN(fast_bn)

    vZ=bn.add(gum.LabelizedVariable('Z','Z',3))
    vX=bn.add(gum.LabelizedVariable('X','X',10))
    vY=bn.add(gum.LabelizedVariable('Y','Y',5))
    vT=bn.add(gum.LabelizedVariable('T','T',10))

    vIvar=bn.add(gum.LabelizedVariable('I'+target_variable,'I'+target_variable,2))

    bn.addArc(vZ,vIvar)

    bn.cpt(vZ)[:] = [0.2,0.3,0.5]

    bn.cpt(vX)[:] = gnrt_mar_prob(10)
    bn.cpt(vY)[:] = gnrt_mar_prob(5)
    bn.cpt(vT)[:] = gnrt_mar_prob(10)

    # o = 1-miss_rate
    bn.cpt(vIvar)[:]=[ [o,miss_rate], [1,0], [o,miss_rate]]

    var_indicator = {target_variable:'I'+target_variable}

    return bn, queries, var_indicator, "fast_bn_mar2_mv_"+target_variable
# ---------------------------------------------------------------------------------------------------------
def mcar_random_bn_generator(fast_bn,miss_rate,target_variable):
    bn=gum.fastBN(fast_bn)

    vIvar=bn.add(gum.LabelizedVariable('I'+target_variable,'I'+target_variable,2))

    o = 1-miss_rate
    bn.cpt(vIvar)[:]=[ o, miss_rate]

    var_indicator = {target_variable:'I'+target_variable}

    return bn, queries, var_indicator, "fast_bn_mcar2_"+target_variable
# ---------------------------------------------------------------------------------------------------------
def fast_mar__random_bn_generator_large_domain_small_bn():
    n = random.randint(5, 10)
    m = random.randint(15, 30)
    bn=gum.fastBN(f"IB{str([0,1])}<-Z{str(sorted([n,m]))}->B{str(sorted([n,m]))}")

    var_indicator = {"B":'IB'}
    target_variable = "B"


    return bn, queries, var_indicator, "fast_mar__random_bn_generator_large_domain_small_bn"
# ---------------------------------------------------------------------------------------------------------
def fast_mar__random_bn_generator_huge_bn_small_domain(n_parents):
    n = 1
    # random.randint(1, 3)
    m = 4
    random.randint(6, 8)
    # n_parents = random.randint(10,12)

    bn_str = ""
    for i in range(n_parents):
        bn_str+= f"IB{str([0,1])}<-Z{str(i)}{str(sorted([n,m]))}->B{str(sorted([n,m]))};"

    my_fast_bn = bn_str[:-1]
    # print(my_fast_bn)
    bn=gum.fastBN(my_fast_bn)

    var_indicator = {"B":'IB'}
    target_variable = "B"

    return bn, queries, var_indicator, f"fast_mar__random_bn_generator_huge_bn_small_domain_{n_parents+2}_nodes"
# ---------------------------------------------------------------------------------------------------------
def fast_mar_B__Zn__IB(B_dom, Zn_dom, nbr_zn):

    bn_str = ""
    for i in range(nbr_zn):
        bn_str+= f"IB{str([0,1])}<-Z{str(i)}{{{'|'.join(map(str, [1,2]))}}}->B{{{'|'.join(map(str, [1,2]))}}};"

    bn=gum.fastBN(bn_str[:-1])
    target_variable = "B"
    var_indicator = {target_variable:'I'+target_variable}

    return bn, queries, var_indicator, f"fast_mar_B__Zn__IB_{nbr_zn+2}_nodes"
# ---------------------------------------------------------------------------------------------------------
# bn=gum.fastBN("A->B<-C->D->E<-F<-A;C->G<-H<-I->J")
# twodbn=gum.fastBN("d0[3]->ct<-at<-a0->b0->bt<-a0->dt[3]<-d0<-c0->ct;c0->at",6)
# bn =gum.fastBN("Age{baby|toddler|kid|teen|adult|old}<-Survived{False|True}->Gender{Female|Male};Siblings{False|True}<-Survived->Parents{False|True}")
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------