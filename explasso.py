##### ExpLasso (The Group Lasso for Design of Experiments)
## tanaken ( Kentaro TANAKA, 2016.02- )

import random
import time
import csv
import copy
import numpy as np
from datetime import datetime
from itertools import chain
from cvxopt import matrix, spmatrix, solvers

####################################################################
### Functions

def gen_main_design_mat(levels_list):
    factors_num = len(levels_list)
    levels_list_rev = copy.deepcopy(levels_list)
    levels_list_rev.reverse()
    main_design_mat = np.array([levels_list_rev[0]])
    for i in range(1,factors_num):
        temp_mat_a = np.repeat(np.array(levels_list_rev[i]), np.shape(main_design_mat)[1])
        temp_mat_b = np.tile(main_design_mat, len(levels_list_rev[i]))
        main_design_mat = np.vstack((temp_mat_a, temp_mat_b))
    # print("\nmain_design_mat:")
    # print(main_design_mat)
    return main_design_mat

def gen_model_mat(main_design_mat, model_list):
    candidates_num = np.shape(main_design_mat)[1]
    if len(model_list) >= 1:
        v = model_list[0]
        if v==[]:
            temp_mat = np.array(np.repeat(1, candidates_num))
        else:
            temp_mat = main_design_mat[v[0],:]
            if len(v) >= 2:
                for i in v[1:(len(v))]:
                    temp_mat = temp_mat * main_design_mat[i,:]
    model_mat = temp_mat
    if len(model_list) >= 2:
        for v in model_list[1:(len(model_list))]:
            if v==[]:
                temp_mat = np.repeat(1, candidates_num)
            else:
                temp_mat = main_design_mat[v[0],:]
                if len(v) >= 2:
                    for i in v[1:(len(v))]:
                        temp_mat = temp_mat * main_design_mat[i,:]
            model_mat = np.vstack((model_mat, temp_mat))
    # print("\nmodel_mat:")
    # print(model_mat)
    return model_mat

def gen_proj_mat(mat):
    pinv_norm_mat = np.linalg.pinv(np.dot(mat.T, mat))
    return np.dot(mat, np.dot(pinv_norm_mat, mat.T))

def gen_lambda_weight_list(main_design_mat, model_list,
                           lambda_scale=10.0, lambda_randomness=0.5):
    model_mat = gen_model_mat(main_design_mat, model_list)
    input_mat = model_mat
    terms_num = np.shape(input_mat)[0]
    candidates_num = np.shape(input_mat)[1]
    lambda_weight_list = [0.]*candidates_num
    if type(lambda_randomness)==bool:
        if lambda_randomness:
            for i in range(0, candidates_num):
                lambda_weight_list[i] = random.uniform(0, 1)
        else:
            subspace_index_list = [0]
            compspace_index_list = list(range(1, candidates_num))
            for i in range(0, terms_num):
                temp_mat = input_mat[:, subspace_index_list]
                temp_mat = gen_proj_mat(temp_mat)
                dist_list = np.repeat(0., len(compspace_index_list))
                for j in range(0, len(compspace_index_list)):
                    temp_vec = (input_mat[:, compspace_index_list[j]]).reshape(terms_num,1)
                    dist_list[j] = np.linalg.norm(np.dot( temp_mat, temp_vec ))
                    dist_list[j] = dist_list[j]*dist_list[j]
                    lambda_weight_list[compspace_index_list[j]] = lambda_weight_list[compspace_index_list[j]] + dist_list[j]
                if i != terms_num-1:
                    temp_index_num = compspace_index_list[np.argmin(dist_list)]
                    subspace_index_list.extend([temp_index_num])
                    compspace_index_list.remove(temp_index_num)
    else:
        subspace_index_list = [0]
        compspace_index_list = list(range(1, candidates_num))
        for i in range(0, terms_num):
            temp_mat = input_mat[:, subspace_index_list]
            temp_mat = gen_proj_mat(temp_mat)
            dist_list = np.repeat(0., len(compspace_index_list))
            for j in range(0, len(compspace_index_list)):
                temp_vec = (input_mat[:, compspace_index_list[j]]).reshape(terms_num,1)
                dist_list[j] = np.linalg.norm(np.dot( temp_mat, temp_vec ))
                dist_list[j] = dist_list[j]*dist_list[j]
                lambda_weight_list[compspace_index_list[j]] = lambda_weight_list[compspace_index_list[j]] + dist_list[j]
            if i != terms_num-1:
                temp_index_num = compspace_index_list[np.argmin(dist_list)]
                subspace_index_list.extend([temp_index_num])
                compspace_index_list.remove(temp_index_num)
        positive_lambda_weight_list = [x for x in lambda_weight_list if x > 0]
        if len(positive_lambda_weight_list)==0:
            min_positive_lambda = 1
        else:
            min_positive_lambda = min(positive_lambda_weight_list)
        lambda_randomness = (max(0.0, min(1.0, lambda_randomness)))
        for i in range(0, candidates_num):
            lambda_weight_list[i] = (1.0-lambda_randomness)*lambda_weight_list[i] + lambda_randomness*random.uniform(0.0, min_positive_lambda)
    lambda_weight_list = [lambda_scale*x for x in lambda_weight_list]
    # print("\nlambda_weight_list:")
    # print(lambda_weight_list)
    return lambda_weight_list

def socp_for_design(main_design_mat, levels_list, model_list, estimation_list,
                    kappa_weight_list, lambda_weight_list, filename):
    # print("estimation_list:")
    # print(estimation_list)
    estimation_eq_index_list = estimation_list[0]
    estimation_pen_index_list = estimation_list[1]
    estimation_pen_weight_list = estimation_list[2]
    estimation_ineq_index_list = estimation_list[3]
    estimation_ineq_tol_list = estimation_list[4]
    estimators_list = list(set(list(chain.from_iterable([estimation_eq_index_list, estimation_pen_index_list, estimation_ineq_index_list]))))
    estimators_list.sort()
    factors_num = len(levels_list)
    terms_num = len(model_list)
    estimators_num = len(estimators_list)
    candidates_num = np.shape(main_design_mat)[1]
    eq_const_num = len(estimation_eq_index_list)
    penalties_num = len(estimation_pen_index_list)
    ineq_const_num = len(estimation_ineq_index_list)
    if estimators_num != len(list(chain.from_iterable([estimation_eq_index_list, estimation_ineq_index_list, estimation_pen_index_list]))):
        print("Error!! 00")
    if len(estimation_ineq_index_list) != len(estimation_ineq_tol_list):
        print("Error!! 10")
    if len(estimation_pen_index_list) != len(estimation_pen_weight_list):
        print("Error!! 20")
    if len(kappa_weight_list) != estimators_num:
        print("Error!! 30")
    if len(lambda_weight_list) != candidates_num:
        print("Error!! 40")
    if max(list(chain.from_iterable(model_list))) >= factors_num:
        print("Error!! 50")
    if estimation_eq_index_list != []:
        if max(estimation_eq_index_list) >= terms_num:
            print("Error!! 60")
    if estimation_pen_index_list != []:
        if max(estimation_pen_index_list) >= terms_num:
            print("Error!! 70")
    if estimation_ineq_index_list != []:
        if max(estimation_ineq_index_list) >= terms_num:
            print("Error!! 80")
    model_mat = gen_model_mat(main_design_mat, model_list)
    print("\nmain_design_mat:")
    print(main_design_mat)
    print("\nmodel_mat:")
    print(model_mat)
    x_var_num = estimators_num * candidates_num
    all_var_num = x_var_num + 2*estimators_num + candidates_num + 2*penalties_num + ineq_const_num
    cvec = [0.]*x_var_num
    cvec.extend([0.]*estimators_num)
    cvec.extend(kappa_weight_list)
    cvec.extend(lambda_weight_list)
    cvec.extend([0.]*penalties_num)
    cvec.extend(estimation_pen_weight_list)
    cvec.extend([0.]*ineq_const_num)
    cvec = matrix(cvec)
    print("kappa_weight_list:")
    print(kappa_weight_list)
    print("lambda_weight_list:")
    print(lambda_weight_list)
    print("estimation_pen_weight_list:")
    print(estimation_pen_weight_list)
    print("cvec:")
    print(cvec)
    Gqmat = []
    hqvec = []
    # L2
    for i in range(0, estimators_num):
        temp_val_list = [-1]+[1]*candidates_num
        temp_row_list = list(range(0,1+candidates_num))
        temp_col_list = [x_var_num+i]+list(range(i*candidates_num, (i+1)*candidates_num))
        Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (candidates_num+1, all_var_num))]
        hqvec += [matrix(spmatrix([], [], [], (candidates_num+1,1)))]
        temp_val_list = [-1, -1, 2]
        temp_row_list = [0, 1, 2]
        temp_index_num = x_var_num+estimators_num+i
        temp_col_list = [temp_index_num, temp_index_num, x_var_num+i]
        Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (3, all_var_num))]
        hqvec += [matrix(spmatrix([1,-1], [0,1], [0,0], (3,1)))]
    # L1
    for i in range(0, candidates_num):
        temp_val_list = [-1]+[1]*estimators_num
        temp_row_list = list(range(0,1+estimators_num))
        temp_col_list = [x_var_num+2*estimators_num+i]+list(range(i, x_var_num, candidates_num))
        Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (estimators_num+1, all_var_num))]
        hqvec += [matrix(spmatrix([], [], [], (estimators_num+1,1)))]
    for i in range(0, penalties_num):
        temp_val_list = [-1]
        temp_row_list = [0]
        temp_col_list = [x_var_num+2*estimators_num+candidates_num+i]
        temp_index_num = estimators_list.index(estimation_pen_index_list[i])
        for j in range(0, terms_num):
            for k in range(0, candidates_num):
                temp_val_list += [model_mat[j,k]]
                temp_row_list += [j+1]
                temp_col_list += [temp_index_num*candidates_num+k]
        Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (terms_num+1, all_var_num))]
        hqvec += [matrix(spmatrix([1], [estimation_pen_index_list[i]+1], [0], (terms_num+1,1)))]
        temp_val_list = [-1, -1, 2]
        temp_row_list = [0, 1, 2]
        temp_index_num = x_var_num+2*estimators_num+candidates_num+penalties_num+i
        temp_col_list = [temp_index_num, temp_index_num,x_var_num+2*estimators_num+candidates_num+i]
        Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (3, all_var_num))]
        hqvec += [matrix(spmatrix([1,-1], [0,1], [0,0], (3,1)))]
    Glmat=matrix([])
    hlvec=matrix([])
    if estimation_ineq_index_list != []:
        for i in range(0, ineq_const_num):
            temp_val_list = [-1]
            temp_row_list = [0]
            temp_col_list = [x_var_num+2*estimators_num+candidates_num+2*penalties_num+i]
            temp_index_num = estimators_list.index(estimation_ineq_index_list[i])
            for j in range(0, terms_num):
                for k in range(0, candidates_num):
                    temp_val_list += [model_mat[j,k]]
                    temp_row_list += [j+1]
                    temp_col_list += [temp_index_num*candidates_num+k]
            Gqmat += [spmatrix(temp_val_list, temp_row_list, temp_col_list, (terms_num+1, all_var_num))]
            hqvec += [matrix(spmatrix([1], [estimation_ineq_index_list[i]+1], [0], (terms_num+1,1)))]
        temp_val_list = []
        temp_row_list = []
        temp_col_list = []
        for i in range(0, ineq_const_num):
            temp_val_list += [1]
            temp_row_list += [i]
            temp_col_list += [x_var_num+2*estimators_num+candidates_num+2*penalties_num+i]
        Glmat = spmatrix(temp_val_list, temp_row_list, temp_col_list, (ineq_const_num, all_var_num))
        hlvec = matrix(estimation_ineq_tol_list)
    print("Glmat:")
    print(matrix(Glmat))
    print("hlvec:")
    print(matrix(hlvec))
    print("Gqmat:")
    print(Gqmat,"\n")
    for i in range(0, estimators_num):
        print(matrix(Gqmat[i]))
    print("hqvec:")
    print(hqvec,"\n")
    for i in range(0, estimators_num):
        print(matrix(hqvec[i]))
    Amat = matrix([])
    bvec = matrix([])
    temp_val_list = []
    temp_row_list = []
    temp_col_list = []
    for i in range(0, eq_const_num):
        temp_index_num = estimators_list.index(estimation_eq_index_list[i])
        for j in range(0, terms_num):
            for k in range(0, candidates_num):
                temp_val_list += [model_mat[j,k]]
                temp_row_list += [i*terms_num+j]
                temp_col_list += [temp_index_num*candidates_num+k]
    Amat = spmatrix(temp_val_list, temp_row_list, temp_col_list, (eq_const_num*terms_num, all_var_num))
    temp_val_list = []
    temp_row_list = []
    temp_col_list = []
    for i in range(0, eq_const_num):
            temp_val_list += [1]
            temp_row_list += [i*terms_num+estimation_eq_index_list[i]]
            temp_col_list += [0]
    bvec = matrix(spmatrix(temp_val_list, temp_row_list, temp_col_list, (eq_const_num*terms_num, 1)))
    print("Amat:")
    print(Amat,"\n")
    print(matrix(Amat))
    print("bvec:")
    print(matrix(bvec))
    filename = filename+".check.csv"
    f = open(filename,"w")
    csvf = csv.writer(f, lineterminator='\n')
    csvf.writerow(["main_design_mat:"])
    temp_mat = np.array(main_design_mat)
    for i in range(0, factors_num):
        csvf.writerow(temp_mat[i,:])
    csvf.writerow(["model_list:"])
    csvf.writerow(model_list)
    csvf.writerow(["model_mat:"])
    temp_mat = np.array(model_mat)
    for i in range(0, terms_num):
        csvf.writerow(temp_mat[i,:])
    csvf.writerow(["estimation_eq_index_list:"])
    csvf.writerow(estimation_eq_index_list)
    csvf.writerow(np.array(model_list, dtype=object)[estimation_eq_index_list])
    csvf.writerow(["estimation_pen_index_list:"])
    csvf.writerow(estimation_pen_index_list)
    csvf.writerow(np.array(model_list, dtype=object)[estimation_pen_index_list])
    csvf.writerow(["estimation_pen_weight_list:"])
    csvf.writerow(estimation_pen_weight_list)
    csvf.writerow(["estimation_ineq_index_list:"])
    csvf.writerow(estimation_ineq_index_list)
    csvf.writerow(np.array(model_list, dtype=object)[estimation_ineq_index_list])
    csvf.writerow(["estimation_ineq_tol_list:"])
    csvf.writerow(estimation_ineq_tol_list)
    csvf.writerow(["kappa_weight_list:"])
    csvf.writerow(kappa_weight_list)
    csvf.writerow(["lambda_weight_list:"])
    csvf.writerow(lambda_weight_list)
    csvf.writerow(["c:"])
    temp_mat = np.array(matrix(cvec))
    for i in range(0, all_var_num):
        csvf.writerow(temp_mat[i])
    csvf.writerow(["Gl:"])
    temp_mat = np.array(matrix(Glmat))
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i,:])
    csvf.writerow(["hl:"])
    temp_mat = np.array(matrix(hlvec))
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i])
    for i in range(0, len(Gqmat)):
        csvf.writerow(["Gq:", i])
        temp_mat = np.array(matrix(Gqmat[i]))
        temp_index_num = np.shape(temp_mat)[0]
        for j in range(0, temp_index_num):
            csvf.writerow(temp_mat[j,:])
        csvf.writerow(["hq:", i])
        temp_mat = np.array(matrix(hqvec[i]))
        temp_index_num = np.shape(temp_mat)[0]
        for j in range(0, temp_index_num):
            csvf.writerow(temp_mat[j])
    csvf.writerow(["Amat:"])
    temp_mat = np.array(matrix(Amat))
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i])
    csvf.writerow(["bvec:"])
    temp_mat = np.array(matrix(bvec))
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i])
    f.close()
    print("\n<<  SOCP  >>")
    if estimation_ineq_index_list != []:
        sol = solvers.socp(c=cvec, Gl=Glmat, hl=hlvec, Gq=Gqmat, hq=hqvec, A=Amat, b=bvec)
    else:
        sol = solvers.socp(c=cvec, Gq=Gqmat, hq=hqvec, A=Amat, b=bvec)
    args = {"c":cvec, "Gl":Glmat, "hl":hlvec, "Gq":Gqmat, "hq":hqvec, "A":Amat, "b":bvec}
    model_mat = np.array(model_mat)
    return (model_mat, args, sol)

def choose_design_points(levels_list, main_design_mat,
                         model_list, model_mat, estimation_list,
                         sol, computation_time, tol, filename):
    estimation_eq_index_list = estimation_list[0]
    estimation_pen_index_list = estimation_list[1]
    estimation_pen_weight_list = estimation_list[2]
    estimation_ineq_index_list = estimation_list[3]
    estimation_ineq_tol_list = estimation_list[4]
    estimators_list = list(set(list(chain.from_iterable([estimation_eq_index_list, estimation_pen_index_list, estimation_ineq_index_list]))))
    estimators_list.sort()
    factors_num = len(levels_list)
    terms_num = len(model_list)
    estimators_num = len(estimators_list)
    candidates_num = np.shape(main_design_mat)[1]
    eq_const_num = len(estimation_eq_index_list)
    penalties_num = len(estimation_pen_index_list)
    ineq_const_num = len(estimation_ineq_index_list)
    x_var_num = estimators_num * candidates_num
    all_var_num = x_var_num + 2*estimators_num + candidates_num + 2*penalties_num + ineq_const_num
    temp_mat_a = np.array(sol["x"])
    temp_mat_b = np.array(model_mat)
    design_points_list = []
    design_mat = np.array([[0.]*terms_num])
    design_mat = design_mat.reshape((terms_num,1))
    for i in range(0, candidates_num):
        if abs(temp_mat_a[x_var_num+2*estimators_num+i]) > tol:
            design_points_list += [i]
            temp_mat_c = temp_mat_b[:,i]
            temp_mat_c = temp_mat_c.reshape((terms_num,1))
            design_mat = np.hstack([design_mat, temp_mat_c])
    design_mat = design_mat[:,1:]
    design_points_num = len(design_points_list)
    print("\ncomputation_time:")
    print(computation_time)
    print("\ntol:")
    print(tol)
    print("\nsol.keys:")
    print(sol.keys())
    print("\nx:")
    print(sol["x"])
    print("\nlasso(L1 norm):")
    temp_mat = np.array(sol["x"])
    print(temp_mat[range(x_var_num+2*estimators_num, x_var_num+2*estimators_num+candidates_num)])
    print("\nSqErr_pen(sqrt):")
    temp_mat = np.array(sol["x"])
    print(temp_mat[range(x_var_num+2*estimators_num+candidates_num, x_var_num+2*estimators_num+candidates_num+penalties_num)])
    print("\nSqErr_ineq(sqrt):")
    temp_mat = np.array(sol["x"])
    print(temp_mat[x_var_num+2*estimators_num+candidates_num+2*penalties_num:])
    print("\ndesign_points_list:")
    print(design_points_list)
    print("\ndesign_mat:")
    print(design_mat)
    print("\n")
    filename = filename+".sol.csv"
    f = open(filename,"w")
    csvf = csv.writer(f, lineterminator='\n')
    csvf.writerow(["computation_time:"])
    csvf.writerow([computation_time])
    csvf.writerow(["tol:"])
    csvf.writerow([tol])
    csvf.writerow(["design_points_list:"])
    csvf.writerow(design_points_list)
    csvf.writerow(["design_mat:"])
    temp_mat = np.array(design_mat)
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i,:])
    csvf.writerow(["x_coef:"])
    temp_mat = np.array(sol["x"])
    for i in range(0, estimators_num):
        csvf.writerow(["x:", model_list[estimators_list[i]]])
        temp_vec = temp_mat[range(i*candidates_num, (i+1)*candidates_num)]
        temp_vec = temp_vec[design_points_list]
        temp_vec = temp_vec.reshape(1, design_points_num)
        csvf.writerow(matrix(temp_vec))
    csvf.writerow(["Var(sqrt):"])
    temp_mat = np.array(sol["x"])
    for i in range(0, estimators_num):
        temp_vec = [model_list[estimators_list[i]]]
        temp_vec.extend(temp_mat[x_var_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["Var:"])
    temp_mat = np.array(sol["x"])
    for i in range(0, estimators_num):
        temp_vec = [model_list[estimators_list[i]]]
        temp_vec.extend(temp_mat[x_var_num+estimators_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["lasso(L1 norm):"])
    temp_mat = np.array(sol["x"])
    for i in range(0, candidates_num):
        temp_vec = [i]
        temp_vec.extend(temp_mat[x_var_num+2*estimators_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["SqErr_pen(sqrt):"])
    temp_mat = np.array(sol["x"])
    for i in range(0, penalties_num):
        temp_vec = [model_list[estimation_pen_index_list[i]]]
        temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["SqErr_pen:"])
    temp_mat = np.array(sol["x"])
    for i in range(0, penalties_num):
        temp_vec = [model_list[estimation_pen_index_list[i]]]
        temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+penalties_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["SqErr_ineq(sqrt):"])
    temp_mat = np.array(sol["x"])
    for i in range(0, ineq_const_num):
        temp_vec = [model_list[estimation_ineq_index_list[i]]]
        temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+2*penalties_num+i])
        csvf.writerow(temp_vec)
    csvf.writerow(["sol.x:"])
    temp_mat = np.array(sol["x"])
    temp_index_num = np.shape(temp_mat)[0]
    for i in range(0, temp_index_num):
        csvf.writerow(temp_mat[i])
    f.close()
    model_mat = np.array(model_mat)
    design_mat = np.array(design_mat)
    return (design_points_list, design_mat)

####################################################################
### Examples

# #### Demo: generating the L4 orthogonal array (Example 4)
# filepath = "./" # "/home" # path to the output file
# t0 = time.time()
# ## setup parameters
# levels_list = [ [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.] ] ## x0,x1,x2: 2-levels(-1 or 1)
# model_list = [ [], [0], [1], [2] ] ## y = a + b0*x0 + b1*x1 + b2*x2 + eps
# estimation_eq_index_list = [1,2,3] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b0,...,b03)
# estimation_pen_index_list = [] ## penalties for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_pen_weight_list = [] ## penalties for quasi-unbiased estimators
# estimation_ineq_index_list = [] ## inequality constraints for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_ineq_tol_list = [] ## inequality constraints for quasi-unbiased estimators (list of tolerances)
# estimation_list = [estimation_eq_index_list, estimation_pen_index_list, estimation_pen_weight_list, estimation_ineq_index_list, estimation_ineq_tol_list]
# len_estimation_index_list = len(estimation_eq_index_list)+len(estimation_pen_index_list)+len(estimation_ineq_index_list)
# kappa_weight_list = [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
# tol = 1.e-6 ## cut off point
# ## optimization
# main_design_mat = gen_main_design_mat(levels_list) ## candidate design points
# lambda_weight_list = gen_lambda_weight_list(main_design_mat, model_list, 10.0, 0.5) ## weights for L1 norms
# filename = filepath+"/explasso.L4."+datetime.now().strftime("%Y%m%d%H%M%S")
# (model_mat, args, sol) = socp_for_design(main_design_mat, levels_list,
#                                          model_list, estimation_list,
#                                          kappa_weight_list, lambda_weight_list,
#                                          filename)
# t1 = time.time()
# computation_time = str(t1-t0)+"[s]"
# (design_points_list, design_mat) = choose_design_points(levels_list, main_design_mat,
#                                                         model_list, model_mat, estimation_list,
#                                                         sol, computation_time, tol, filename)

#### Demo: generating the L8 orthogonal array (Example 5)
filepath = "./" # "/home" # path to the output file
t0 = time.time()
## setup parameters
levels_list = [ [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.] ] ## [[-1,1],[-1,1],[-1,1],[-1,1]] <=> x0,x1,x2,x3: 2-levels(-1 or 1)
model_list = [ [], [0], [1], [2], [3], [0,1], [0,2], [0,3] ] ## y = a + b0*x0 + b1*x1 + b2*x2 + b3*x3 + b01*x0*x1 + b02*x0*x2 + b03*x0*x3 + eps ## [ ],[0],[1],[2],[3],[0,1],[0,2],[0,3] <=> y=b[ ]+b[0]*x0+b[1]*x1+b[2]*x2+b[3]*x3+b[0,1]*x0*x1+b[0,2]*x0*x2+b[0,3]*x0*x3
estimation_eq_index_list = [1,2,3,4,5,6,7] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b0,...,b03)
estimation_pen_index_list = [] ## penalties for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
estimation_pen_weight_list = [] ## penalties for quasi-unbiased estimators
estimation_ineq_index_list = [] ## inequality constraints for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
estimation_ineq_tol_list = [] ## inequality constraints for quasi-unbiased estimators (list of tolerances)
estimation_list = [estimation_eq_index_list, estimation_pen_index_list, estimation_pen_weight_list, estimation_ineq_index_list, estimation_ineq_tol_list]
len_estimation_index_list = len(estimation_eq_index_list)+len(estimation_pen_index_list)+len(estimation_ineq_index_list)
kappa_weight_list = [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
tol = 1.e-6 ## cut off point
## optimization
main_design_mat = gen_main_design_mat(levels_list) ## candidate design points
lambda_weight_list = gen_lambda_weight_list(main_design_mat, model_list, 1., 0.5) ## weights for L1 norms
filename = filepath+"/explasso.L8."+datetime.now().strftime("%Y%m%d%H%M%S")
(model_mat, args, sol) = socp_for_design(main_design_mat, levels_list,
                                         model_list, estimation_list,
                                         kappa_weight_list, lambda_weight_list,
                                         filename)
t1 = time.time()
computation_time = str(t1-t0)+"[s]"
(design_points_list, design_mat) = choose_design_points(levels_list, main_design_mat,
                                                        model_list, model_mat, estimation_list,
                                                        sol, computation_time, tol, filename)

# #### Demo: generating the L8+ array (Example 6)
# filepath = "./" # "/home" # path to the output file
# t0 = time.time()
# ## setup parameters
# levels_list = [ [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.] ] ## x0,...,x3: 2-levels(-1 or 1)
# model_list = [ [], [0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2] ] ## y = a + b0*x0 + b1*x1 + b2*x2 + b3*x3 + b01*x0*x1 + b02*x0*x2 + b03*x0*x3 + eps
# estimation_eq_index_list = [1,2,3,4,5,6,7,8] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b0,...,b03)
# estimation_pen_index_list = [] ## penalties for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_pen_weight_list = [] ## penalties for quasi-unbiased estimators
# estimation_ineq_index_list = [] ## inequality constraints for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_ineq_tol_list = [] ## inequality constraints for quasi-unbiased estimators (list of tolerances)
# estimation_list = [estimation_eq_index_list, estimation_pen_index_list, estimation_pen_weight_list, estimation_ineq_index_list, estimation_ineq_tol_list]
# len_estimation_index_list = len(estimation_eq_index_list)+len(estimation_pen_index_list)+len(estimation_ineq_index_list)
# kappa_weight_list = [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
# tol = 1.e-6 ## cut off point
# ## optimization
# main_design_mat = gen_main_design_mat(levels_list) ## candidate design points
# lambda_weight_list = gen_lambda_weight_list(main_design_mat, model_list, 10., 0.5) ## weights for L1 norms
# filename = filepath+"/explasso.L8plus."+datetime.now().strftime("%Y%m%d%H%M%S")
# (model_mat, args, sol) = socp_for_design(main_design_mat, levels_list,
#                                          model_list, estimation_list,
#                                          kappa_weight_list, lambda_weight_list,
#                                          filename)
# t1 = time.time()
# computation_time = str(t1-t0)+"[s]"
# (design_points_list, design_mat) = choose_design_points(levels_list, main_design_mat,
#                                                         model_list, model_mat, estimation_list,
#                                                         sol, computation_time, tol, filename)

# #### Demo: 1-10 factors having 2 levels
# filepath = "./" # "/home" # path to the output file
# t0 = time.time()
# ## setup parameters
# levels_list = [ [1., -1.] ] ## [ [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## x0,...,x?: 2-levels(1 or -1)
# model_list =  [ [], [0]] ## [ [], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] ## y = a + b0*x0 + b1*x1 + b2*x2 + ... + b?*x? + eps
# estimation_eq_index_list = [1] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b0,...,b03)
# estimation_pen_index_list = [] ## penalties for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_pen_weight_list = [] ## penalties for quasi-unbiased estimators
# estimation_ineq_index_list = [] ## inequality constraints for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_ineq_tol_list = [] ## inequality constraints for quasi-unbiased estimators (list of tolerances)
# estimation_list = [estimation_eq_index_list, estimation_pen_index_list, estimation_pen_weight_list, estimation_ineq_index_list, estimation_ineq_tol_list]
# len_estimation_index_list = len(estimation_eq_index_list)+len(estimation_pen_index_list)+len(estimation_ineq_index_list)
# kappa_weight_list = [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
# tol = 1.e-6 ## cut off point
# ## optimization
# main_design_mat = gen_main_design_mat(levels_list) ## candidate design points
# lambda_weight_list = gen_lambda_weight_list(main_design_mat, model_list, 10., 0.5) ## weights for L1 norms
# filename = filepath+"/explasso.1-10fators2levels."+datetime.now().strftime("%Y%m%d%H%M%S")
# (model_mat, args, sol) = socp_for_design(main_design_mat, levels_list,
#                                          model_list, estimation_list,
#                                          kappa_weight_list, lambda_weight_list,
#                                          filename)
# t1 = time.time()
# computation_time = str(t1-t0)+"[s]"
# (design_points_list, design_mat) = choose_design_points(levels_list, main_design_mat,
#                                                         model_list, model_mat, estimation_list,
#                                                         sol, computation_time, tol, filename)

# #### Demonstration of the use of 'estimation_*_index_list's
# filepath = "./" # "/home" # path to the output file
# t0 = time.time()
# ## setup parameters
# levels_list = [ [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.] ] ## [[-1,1],[-1,1],[-1,1],[-1,1]] <=> x0,x1,x2,x3: 2-levels(-1 or 1)
# model_list = [ [], [0], [1], [2], [3], [0,1], [0,2], [0,3] ] ## y = a + b0*x0 + b1*x1 + b2*x2 + b3*x3 + b01*x0*x1 + b02*x0*x2 + b03*x0*x3 + eps ## [ ],[0],[1],[2],[3],[0,1],[0,2],[0,3] <=> y=b[ ]+b[0]*x0+b[1]*x1+b[2]*x2+b[3]*x3+b[0,1]*x0*x1+b[0,2]*x0*x2+b[0,3]*x0*x3
# estimation_eq_index_list = [0] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b0,...,b03)
# estimation_pen_index_list = [1, 2, 3, 4] ## penalties for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_pen_weight_list = [0.01, 0.01, 0.01, 0.01] ## penalties for quasi-unbiased estimators
# estimation_ineq_index_list = [5, 6, 7] ## inequality constraints for quasi-unbiased estimators ('estimation_*_index_list's must be mutually disjoint)
# estimation_ineq_tol_list = [0.2, 0.2, 0.2] ## inequality constraints for quasi-unbiased estimators (list of tolerances)
# estimation_list = [estimation_eq_index_list, estimation_pen_index_list, estimation_pen_weight_list, estimation_ineq_index_list, estimation_ineq_tol_list]
# len_estimation_index_list = len(estimation_eq_index_list)+len(estimation_pen_index_list)+len(estimation_ineq_index_list)
# kappa_weight_list = [1.0, 0.02,0.02,0.02,0.02, 0.1,0.1,0.1] # [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
# tol = 1.e-6 ## cut off point
# ## optimization
# main_design_mat = gen_main_design_mat(levels_list) ## candidate design points
# lambda_weight_list = gen_lambda_weight_list(main_design_mat, model_list, 10., 0.5) ## weights for L1 norms
# ## lambda_weight_list = [0.,61.4626437,61.4626437,0.,61.4626437,0.,0.,61.4626437,61.4626437,20.,40.,61.4626437,60.,61.4626437,61.4626437,80.]
# filename = filepath+"/explasso.L8."+datetime.now().strftime("%Y%m%d%H%M%S")
# (model_mat, args, sol) = socp_for_design(main_design_mat, levels_list,
#                                          model_list, estimation_list,
#                                          kappa_weight_list, lambda_weight_list,
#                                          filename)
# t1 = time.time()
# computation_time = str(t1-t0)+"[s]"
# (design_points_list, design_mat) = choose_design_points(levels_list, main_design_mat,
#                                                         model_list, model_mat, estimation_list,
#                                                         sol, computation_time, tol, filename)
