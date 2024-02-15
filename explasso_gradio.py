##### ExpLasso (The Group Lasso for Design of Experiments)
## tanaken ( Kentaro TANAKA, 2016.02- )

import random
import time
import datetime
import copy
import numpy as np
import pandas as pd
from itertools import chain
from cvxopt import matrix, spmatrix, solvers
import gradio as gr

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

def socp_for_design_in_gradio(levels_list, main_design_mat,
                              model_list, estimation_list,
                              kappa_weight_list, lambda_weight_list,
                              show_info=False):
    estimation_eq_index_list = estimation_list[0]
    estimation_pen_index_list = estimation_list[1]
    estimation_pen_weight_list = estimation_list[2]
    estimation_ineq_index_list = estimation_list[3]
    estimation_ineq_tol_list = estimation_list[4]
    estimators_list = list(set(list(chain.from_iterable([estimation_eq_index_list,
                                                         estimation_pen_index_list,
                                                         estimation_ineq_index_list]))))
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
    if show_info:
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
    if show_info:
        print("kappa_weight_list")
        print(kappa_weight_list)
        print("lambda_weight_list")
        print(lambda_weight_list)
        print("estimation_pen_weight_list")
        print(estimation_pen_weight_list)
        print("cvec")
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
    if show_info:
        print("Glmat")
        print(matrix(Glmat))
        print("hlvec")
        print(matrix(hlvec))
        print("Gqmat")
        print(Gqmat,"\n")
        for i in range(0, estimators_num):
            print(matrix(Gqmat[i]))
        print("hqvec")
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
    if show_info:
        print("Amat")
        print(Amat,"\n")
        print(matrix(Amat))
        print("bvec")
        print(matrix(bvec))
        print("\n<<  SOCP  >>")
    if estimation_ineq_index_list != []:
        sol = solvers.socp(c=cvec, Gl=Glmat, hl=hlvec, Gq=Gqmat, hq=hqvec, A=Amat, b=bvec)
    else:
        sol = solvers.socp(c=cvec, Gq=Gqmat, hq=hqvec, A=Amat, b=bvec)
    args = {"c":cvec, "Gl":Glmat, "hl":hlvec, "Gq":Gqmat, "hq":hqvec, "A":Amat, "b":bvec}
    model_mat = np.array(model_mat)
    return (model_mat, args, sol)

def choose_design_points_in_gradio(levels_list, main_design_mat,
                                   model_list, model_mat,
                                   estimation_list, sol,
                                   computation_time, tol,
                                   show_info=False):
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
    temp_mat_a = np.array(sol['x'])
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
    if show_info:
        print("\ncomputation_time:")
        print(computation_time)
        print("\ntol:")
        print(tol)
        print("\nsol.keys:")
        print(sol.keys())
        print("\nx:")
        print(sol['x'])
        print("\nlasso(L1 norm):")
        temp_mat = np.array(sol['x'])
        print(temp_mat[range(x_var_num+2*estimators_num, x_var_num+2*estimators_num+candidates_num)])
        print("\nSqErr_pen(sqrt):")
        temp_mat = np.array(sol['x'])
        print(temp_mat[range(x_var_num+2*estimators_num+candidates_num, x_var_num+2*estimators_num+candidates_num+penalties_num)])
        print("\nSqErr_ineq(sqrt):")
        temp_mat = np.array(sol['x'])
        print(temp_mat[x_var_num+2*estimators_num+candidates_num+2*penalties_num:])
        print("\ndesign_points_list:")
        print(design_points_list)
        print("\ndesign_mat:")
        print(design_mat)
        print("\n")
    model_mat = np.array(model_mat)
    design_mat = np.array(design_mat)
    return (design_points_list, design_mat)
    
def remove_unnecessary_characters(txt):
    txt = txt.replace(" ", "")
    txt = txt.replace("\n", "")
    txt = txt.replace("\t", "")
    txt = txt.replace(",,", ",")
    txt = txt.replace("[[", "[")
    txt = txt.replace("]]", "]")
    txt = txt.replace("[,[", "[")
    txt = txt.replace("],]", "]")
    txt = txt.removeprefix(",")
    txt = txt.removesuffix(",")
    txt = txt.removeprefix("[")
    txt = txt.removesuffix("]")
    txt = txt.removeprefix(",")
    txt = txt.removesuffix(",")
    return txt

def str_to_int_list(txt):
    txt = remove_unnecessary_characters(txt)
    if len(txt) > 0:
        txt = txt.split(",")
        txt = [int(e) for e in txt]
        return txt
    else:
        return []

def str_to_float_list(txt):
    if len(txt) > 0:
        txt = remove_unnecessary_characters(txt)
        txt = txt.split(",")
        txt = [float(e) for e in txt]
        return txt
    else:
        return []

def str_to_int_list_list(txt):
    txt = remove_unnecessary_characters(txt)
    txt = txt.split("],[")
    for i, ar in enumerate(txt):
        ar = ar.split(",")
        if len(ar) == 1:
            if len(ar[0]) > 0:
                ar = [int(ar[0])]
            else:
                ar = []
        else:
            ar = [int(e) for e in ar]
        txt[i] = ar
    return txt

def str_to_float_list_list(txt):
    txt = remove_unnecessary_characters(txt)
    txt = txt.split("],[")
    for i, ar in enumerate(txt):
        ar = ar.split(",")
        if len(ar) == 1:
            if len(ar[0]) > 0:
                ar = [float(ar[0])]
            else:
                ar = []
        else:
            ar = [float(e) for e in ar]
        txt[i] = ar
    return txt

class GradioParameters:
    levels_list = [ [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.] ] # x0,...,x3: 2-levels(-1 or 1)
    model_list = [ [], [0], [1], [2], [3], [0,1], [0,2], [0,3] ] ## y = b[] + b[0]*x0 + b[1]*x1 + b[2]*x2 + b[3]*x3 + b[0,1]*x0*x1 + b[0,2]*x0*x2 + b[0,3]*x0*x3 + eps
    main_design_mat = None
    model_mat = None
    estimation_eq_index_list = [1,2,3,4,5,6,7] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b[],...,b[0,3])
    estimation_pen_index_list = []
    estimation_pen_weight_list = []
    estimation_ineq_index_list = []
    estimation_ineq_tol_list = []
    kappa_weight_list = []
    lambda_weight_list = []
    lambda_scale = 10.0
    lambda_randomness = 0.5
    tol = 1.e-6
    sol = None
    design_points_list = None
    design_mat = None
    computation_time = None
    all_df = pd.DataFrame(
        (np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                    1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
                    1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,
                    1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,
                    1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,
                    1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
                    1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
                    1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1,])).reshape(8, 16),
                    index=["[]", "[0]", "[1]", "[2]", "[3]", "[0,1]", "[0,2]", "[0,3]"],
                    columns=["No. 00", "No. 01", "No. 02", "No. 03", "No. 04", "No. 05", "No. 06", "No. 07",
                             "No. 08", "No. 09", "No. 10", "No. 11", "No. 10", "No. 13", "No. 14", "No. 15"])
    output_df = pd.DataFrame(
        (np.array([ 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, -1, -1, -1, -1,
                    1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, 1, -1, 1, -1, 1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    1, 1, -1, -1, -1, -1, 1, 1,
                    1, -1, 1, -1, -1, 1, -1, 1,
                    1, -1, -1, 1, -1, 1, 1, -1])).reshape(8, 8),
                    index=["[]", "[0]", "[1]", "[2]", "[3]", "[0,1]", "[0,2]", "[0,3]"],
                    columns=["No. 00", "No. 03", "No. 05", "No. 06", "No. 08", "No. 11", "No. 13", "No. 14"])
    output_design_mat = (np.array([ 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, -1, -1, -1, -1,
                    1, 1, -1, -1, 1, 1, -1, -1,
                    1, -1, 1, -1, 1, -1, 1, -1,
                    1, -1, -1, 1, 1, -1, -1, 1,
                    1, 1, -1, -1, -1, -1, 1, 1,
                    1, -1, 1, -1, -1, 1, -1, 1,
                    1, -1, -1, 1, -1, 1, 1, -1])).reshape(8, 8)
    output_data = "[0, 3, 5, 6, 8, 11, 13, 14]"
    output_filename = "output_explasso.csv"
    enumeration_filename = "enumeration.csv"
    all_information_filename = "info.csv"
    # all_information = None
   
def start_explasso(levels_text, model_text, 
                   eq_index_text,
                   lambda_scale_slider, lambda_randomness_slider,
                   pen_index_text, pen_weight_text,
                   ineq_index_text, ineq_tol_text,
                   kappa_weight_text, lambda_weight_text):
    t0 = time.time()
    ## setup parameters
    GradioParameters.levels_list = str_to_float_list_list(levels_text) ## example: [ [1., -1.], [1., -1.], [1., -1.], [1., -1.] ] ## [ [-1., 1.], [-1., 1.], [-1., 1.], [-1., 1.] ] # x0,...,x3: 2-levels(-1 or 1)
    GradioParameters.model_list = str_to_int_list_list(model_text) ## example: [ [], [0], [1], [2], [3], [0,1], [0,2], [0,3] ]  ## y = b[] + b[0]*x0 + b[1]*x1 + b[2]*x2 + b[3]*x3 + b[0,1]*x0*x1 + b[0,2]*x0*x2 + b[0,3]*x0*x3 + eps
    GradioParameters.estimation_eq_index_list = str_to_int_list(eq_index_text) ## example: [1,2,3,4,5,6,7] ## equality constraints for unbiased estimators ('estimation_*_index_list's must be mutually disjoint), (b[],...,b[0,3])
    GradioParameters.estimation_pen_index_list = str_to_int_list(pen_index_text) ## penalties for unbiased estimators ([estimation_*_index_list]s must be mutually disjoint)
    GradioParameters.estimation_pen_weight_list = str_to_float_list(pen_weight_text) ## penalties for unbiased estimators
    GradioParameters.estimation_ineq_index_list = str_to_int_list(ineq_index_text) ## inequality constraints for unbiased estimators ([estimation_*_index_list]s must be mutually disjoint)
    GradioParameters.estimation_ineq_tol_list = str_to_float_list(ineq_tol_text) ## inequality constraints for unbiased estimators
    GradioParameters.kappa_weight_list = str_to_float_list(kappa_weight_text) ## the weight(s) for variances of estimators (Example: [1.,1.,1.,1.,1.,1.,1.])
    GradioParameters.lambda_weight_list = str_to_float_list(lambda_weight_text) ## the weight(s) for L1 regularization for all enumerated trial(s)
    GradioParameters.lambda_scale = float(lambda_scale_slider)
    GradioParameters.lambda_randomness = float(lambda_randomness_slider)
    estimation_list = [GradioParameters.estimation_eq_index_list, 
                       GradioParameters.estimation_pen_index_list, GradioParameters.estimation_pen_weight_list,
                       GradioParameters.estimation_ineq_index_list, GradioParameters.estimation_ineq_tol_list]
    len_estimation_index_list = len(GradioParameters.estimation_eq_index_list)+len(GradioParameters.estimation_pen_index_list)+len(GradioParameters.estimation_ineq_index_list)
    if len(GradioParameters.kappa_weight_list)!=len_estimation_index_list:
        GradioParameters.kappa_weight_list = [1.]*(len_estimation_index_list) ## weights for the squared losses of unbiased estimators
    GradioParameters.tol = 1.e-6 ## cut off point
    ## optimization
    GradioParameters.main_design_mat = gen_main_design_mat(GradioParameters.levels_list) ## candidate design points
    (main_design_mat_nrows, main_design_mat_ncols) = GradioParameters.main_design_mat.shape
    if len(GradioParameters.lambda_weight_list)!=main_design_mat_ncols:
        GradioParameters.lambda_weight_list = gen_lambda_weight_list(GradioParameters.main_design_mat, GradioParameters.model_list,
                                                    GradioParameters.lambda_scale,
                                                    GradioParameters.lambda_randomness) ## weights for L1 norms
    (GradioParameters.model_mat, args, GradioParameters.sol) = socp_for_design_in_gradio(GradioParameters.levels_list, 
                                                                        GradioParameters.main_design_mat,
                                                                        GradioParameters.model_list,
                                                                        estimation_list, 
                                                                        GradioParameters.kappa_weight_list,
                                                                        GradioParameters.lambda_weight_list)
    t1 = time.time()
    GradioParameters.computation_time = str(t1-t0) # +"[s]"
    (GradioParameters.design_points_list, GradioParameters.design_mat) = choose_design_points_in_gradio(GradioParameters.levels_list,
                                                                      GradioParameters.main_design_mat, 
                                                                      GradioParameters.model_list, 
                                                                      GradioParameters.model_mat, 
                                                                      estimation_list, GradioParameters.sol, 
                                                                      GradioParameters.computation_time,
                                                                      GradioParameters.tol)
    # print("Fully-Enumerated model matrix:")
    # print(model_mat.transpose())
    print("Lambda weights list:")
    print(GradioParameters.lambda_weight_list)
    print("Design matrix:")
    print(GradioParameters.design_mat.transpose())
    len_zfill = len(str(main_design_mat_ncols))
    GradioParameters.all_df = pd.DataFrame(data=GradioParameters.model_mat,
                                            index=[str(e) for e in GradioParameters.model_list],
                                            columns=["No. "+str(i).zfill(len_zfill) for i in range(main_design_mat_ncols)])
    GradioParameters.output_data = GradioParameters.design_points_list
    GradioParameters.output_design_mat = GradioParameters.design_mat
    GradioParameters.output_df = pd.DataFrame(data=GradioParameters.output_design_mat,
                                              index=[str(e) for e in GradioParameters.model_list],
                                              columns=["No. "+str(i).zfill(len_zfill) for i in GradioParameters.output_data])
    output_text_value = str(len(GradioParameters.output_data))
    output_text_value = output_text_value + " trial(s) "+str(GradioParameters.output_data)
    output_text_value = output_text_value + " were selected from a total of "+str(main_design_mat_ncols)+" trial(s)."
    output_df_value = GradioParameters.output_df.transpose().reset_index()
    return [gr.Textbox(value=output_text_value),
            gr.Dataframe(value=output_df_value),
            gr.File(interactive=False, visible=False),
            gr.Button(value="Show \"All enumerated trial(s)\""),
            gr.Dataframe(value=None, visible=False),
            gr.Button(value="", visible=False),
            gr.File(interactive=False, visible=False),
            gr.File(interactive=False, visible=False)
            ]

def export_design_mat_csv():
    GradioParameters.output_filename = "design_matrix_explasso." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
    GradioParameters.output_df.transpose().to_csv(GradioParameters.output_filename)
    return gr.File(value=GradioParameters.output_filename, visible=True)

def show_all_pattern(all_patterns_button):
    if all_patterns_button == "Show \"All enumerated trial(s)\"":
        return (gr.Button(value="Hide \"All enumerated trial(s)\""),
                gr.Dataframe(value=GradioParameters.all_df.transpose().reset_index(), visible=True),
                gr.Button(value="Export \"All enumerated trial(s)\"", visible=True),
                gr.File(interactive=False, visible=False))
    else:
        return (gr.Button(value="Show \"All enumerated trial(s)\""),
                gr.Dataframe(value=None, visible=False),
                gr.Button(value="", visible=False),
                gr.File(interactive=False, visible=False))

def export_all_patterns_csv():
    GradioParameters.enumeration_filename = "all_enumerated_trials_explasso." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
    GradioParameters.all_df.transpose().to_csv(GradioParameters.enumeration_filename)
    return gr.File(value=GradioParameters.enumeration_filename, visible=True)

def export_all_information_csv():
    import csv
    if GradioParameters.sol is None:
        return gr.File(interactive=False, visible=False)
    else:
        GradioParameters.all_information_filename = "output_explasso." + (datetime.datetime.now()).strftime('%Y%m%d%H%M%S') + ".csv"
        filename = GradioParameters.all_information_filename
        estimators_list = list(set(list(chain.from_iterable([GradioParameters.estimation_eq_index_list,
                                                            GradioParameters.estimation_pen_index_list,
                                                            GradioParameters.estimation_ineq_index_list]))))
        estimators_list.sort()
        factors_num = len(GradioParameters.levels_list)
        terms_num = len(GradioParameters.model_list)
        estimators_num = len(estimators_list)
        candidates_num = np.shape(GradioParameters.main_design_mat)[1]
        len_zfill = len(str(candidates_num))
        eq_const_num = len(GradioParameters.estimation_eq_index_list)
        penalties_num = len(GradioParameters.estimation_pen_index_list)
        ineq_const_num = len(GradioParameters.estimation_ineq_index_list)
        x_var_num = estimators_num * candidates_num
        # all_var_num = x_var_num + 2*estimators_num + candidates_num + 2*penalties_num + ineq_const_num
        f = open(filename,"w")
        csvf = csv.writer(f, lineterminator='\n')
        csvf.writerow(["All enumerated trial(s) (transposed):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in range(candidates_num)])
        temp_mat = np.array(GradioParameters.main_design_mat)
        for i in range(0, factors_num):
            csvf.writerow(["["+str(i)+"]"] + list(temp_mat[i,:]))
        csvf.writerow([""])
        csvf.writerow(["Model:"])
        csvf.writerow([""] + GradioParameters.model_list)
        csvf.writerow([""])
        csvf.writerow(["Model matrix for all enumerated trial(s) (transposed):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in range(candidates_num)])
        temp_mat = np.array(GradioParameters.model_mat)
        for i in range(0, terms_num):
            csvf.writerow([GradioParameters.model_list[i]] + list(temp_mat[i,:]))
        csvf.writerow([""])
        csvf.writerow(["Index(es) of variable(s) in Model to be unbiasedly estimated:"])
        if len(GradioParameters.estimation_eq_index_list)==0:
            csvf.writerow([""] + ["None"])
        else:
            csvf.writerow([""] + GradioParameters.estimation_eq_index_list)
            csvf.writerow([""] + list(np.array(GradioParameters.model_list, dtype=object)[GradioParameters.estimation_eq_index_list]))
        csvf.writerow([""])
        csvf.writerow(["Index(es) of quasi-unbiased estimator(s) with penalty(ies):"])
        if len(GradioParameters.estimation_pen_index_list)==0:
            csvf.writerow([""] + ["None"])
            csvf.writerow([""])
            csvf.writerow(["Penalty(ies) of quasi-unbiased estimator(s) with penalty(ies):"])
            csvf.writerow([""] + ["None"])
        else:
            csvf.writerow([""] + GradioParameters.estimation_pen_index_list)
            csvf.writerow([""] + list(np.array(GradioParameters.model_list, dtype=object)[GradioParameters.estimation_pen_index_list]))
            csvf.writerow([""])
            csvf.writerow(["Penalty(ies) of quasi-unbiased estimator(s) with penalty(ies):"])
            csvf.writerow([""] + GradioParameters.estimation_pen_weight_list)
        csvf.writerow([""])
        csvf.writerow(["Index(es) of quasi-unbiased estimator(s) with tolerance inequality(ies):"])
        if len(GradioParameters.estimation_ineq_index_list)==0:
            csvf.writerow([""] + ["None"])
            csvf.writerow([""])
            csvf.writerow(["Tolerance(s) of quasi-unbiased estimator(s) with tolerance inequality(ies):"])
            csvf.writerow([""] + ["None"])
        else:
            csvf.writerow([""] + GradioParameters.estimation_ineq_index_list)
            csvf.writerow([""] + list(np.array(GradioParameters.model_list, dtype=object)[GradioParameters.estimation_ineq_index_list]))
            csvf.writerow([""])
            csvf.writerow(["Tolerance(s) of quasi-unbiased estimator(s) with tolerance inequality(ies):"])
            csvf.writerow([""] + GradioParameters.estimation_ineq_tol_list)
        csvf.writerow([""])
        csvf.writerow(["Weight(s) for variance(s) of estimator(s):"])
        csvf.writerow([""] + [GradioParameters.model_list[i] for i in estimators_list])
        csvf.writerow([""] + GradioParameters.kappa_weight_list)
        csvf.writerow([""])
        csvf.writerow(["Weight(s) for L1 regularization for all enumerated trial(s):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in range(candidates_num)])
        csvf.writerow([""] + GradioParameters.lambda_weight_list)
        csvf = csv.writer(f, lineterminator='\n')
        design_points_num = len(GradioParameters.design_points_list)
        csvf.writerow([""])
        csvf.writerow(["Computation time[s]:"])
        csvf.writerow([""] + [GradioParameters.computation_time])
        # csvf.writerow([""])
        # csvf.writerow(["tol"])
        # csvf.writerow([GradioParameters.tol])
        csvf.writerow([""])
        csvf.writerow(["Selected trial(s):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in GradioParameters.design_points_list])
        csvf.writerow([""])
        csvf.writerow(["Design matrix (transposed):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in GradioParameters.design_points_list])
        temp_mat = np.array(GradioParameters.design_mat)
        temp_index_num = np.shape(temp_mat)[0]
        for i in range(0, temp_index_num):
            csvf.writerow([GradioParameters.model_list[i]] + list(temp_mat[i,:]))
        csvf.writerow([""])
        csvf.writerow(["Coefficient(s) of estimator(s):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in GradioParameters.design_points_list])
        temp_mat = np.array(GradioParameters.sol["x"])
        for i in range(0, estimators_num):
            temp_vec = temp_mat[range(i*candidates_num, (i+1)*candidates_num)]
            temp_vec = temp_vec[GradioParameters.design_points_list]
            temp_vec = temp_vec.reshape(1, design_points_num)
            csvf.writerow([GradioParameters.model_list[estimators_list[i]]] + list(matrix(temp_vec)))
        # csvf.writerow([""])
        # csvf.writerow(["Variance(s) (sqrt)"])
        # temp_mat = np.array(GradioParameters.sol["x"])
        # for i in range(0, estimators_num):
        #     temp_vec = [GradioParameters.model_list[estimators_list[i]]]
        #     temp_vec.extend(temp_mat[x_var_num+i])
        #     csvf.writerow(temp_vec)
        csvf.writerow([""])
        csvf.writerow(["Variance(s) of estimator(s):"])
        temp_mat = np.array(GradioParameters.sol["x"])
        for i in range(0, estimators_num):
            temp_vec = [GradioParameters.model_list[estimators_list[i]]]
            temp_vec.extend(temp_mat[x_var_num+estimators_num+i])
            csvf.writerow(temp_vec)
        # csvf.writerow([""])
        # csvf.writerow(["SqErr_pen(sqrt)":])
        # temp_mat = np.array(GradioParameters.sol["x"])
        # for i in range(0, penalties_num):
        #     temp_vec = [GradioParameters.model_list[GradioParameters.estimation_pen_index_list[i]]]
        #     temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+i])
        #     csvf.writerow(temp_vec)
        # csvf.writerow([""])
        # csvf.writerow(["SqErr_pen":])
        # temp_mat = np.array(GradioParameters.sol["x"])
        # for i in range(0, penalties_num):
        #     temp_vec = [GradioParameters.model_list[GradioParameters.estimation_pen_index_list[i]]]
        #     temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+penalties_num+i])
        #     csvf.writerow(temp_vec)
        # csvf.writerow([""])
        # csvf.writerow(["SqErr_ineq(sqrt):"])
        # temp_mat = np.array(GradioParameters.sol["x"])
        # for i in range(0, ineq_const_num):
        #     temp_vec = [GradioParameters.model_list[GradioParameters.estimation_ineq_index_list[i]]]
        #     temp_vec.extend(temp_mat[x_var_num+2*estimators_num+candidates_num+2*penalties_num+i])
        #     csvf.writerow(temp_vec)
        csvf.writerow([""])
        csvf.writerow(["Values(s) of coefficient(s) for all enumerated trial(s):"])
        csvf.writerow([""] + ["No. "+str(i).zfill(len_zfill) for i in range(candidates_num)])
        temp_mat = np.array(GradioParameters.sol["x"])
        temp_vec = []
        for i in range(0, candidates_num):
            temp_vec.append( float((temp_mat[x_var_num+2*estimators_num+i])[0]) )
        csvf.writerow([""] + temp_vec)
        # csvf.writerow([""])
        # csvf.writerow(["sol.x:"])
        # temp_mat = np.array(GradioParameters.sol["x"])
        # temp_index_num = np.shape(temp_mat)[0]
        # for i in range(0, temp_index_num):
        #     csvf.writerow(temp_mat[i])
        f.close()
        return gr.File(value=GradioParameters.all_information_filename, visible=True)

with gr.Blocks(title="Explasso") as demo:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                levels_text = gr.Textbox(label="Levels", interactive=True,
                                      value="[1., -1.], [1., -1.], [1., -1.], [1., -1.]",
                                      info="Example: [1,-1],[1,-1],[1,-1],[1,-1] <=> x0,x1,x2,x3: two-levels(1 or -1)")
                model_text = gr.Textbox(label="Model", interactive=True,
                                     value="[ ], [0], [1], [2], [3], [0,1], [0,2], [0,3]",
                                     info="Example: [ ],[0],[1],[2],[3],[0,1],[0,2],[0,3] <=> y=b[]+b[0]*x0+b[1]*x1+b[2]*x2+b[3]*x3+b[0,1]*x0*x1+b[0,2]*x0*x2+b[0,3]*x0*x3")
                eq_index_text = gr.Textbox(label="Index(es) of variable(s) in Model to be unbiasedly estimated", interactive=True,
                                        value="1,2,3,4",
                                        info="Choose from 0 to length(Model)-1 (Example: [1,2,3,4] <=> Unbiased estimation for b[0],b[1],b[2],b[3])")
                lambda_scale_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.000001, value=10.0, 
                                    label="Scale of weight(s) for L1 regularization",
                                    info="Set scale of weight(s) for L1 regularization for all enumerated trial(s) in the range of 0.0 to 10000.0", interactive=True)
                lambda_randomness_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, value=0.5, 
                                    label="Randomness of weight(s) for L1 regularization",
                                    info="Set randomness of weight(s) for L1 regularization for all enumerated trial(s) in the range of 0.0 to 1.0", interactive=True)
                with gr.Accordion(label="Advanced Settings", open=False):
                    gr.Markdown(value="Settings of quasi-unbiased estimator(s) (Note: Indexes in the three index entry fields must be mutually disjoint)")
                    pen_index_text = gr.Textbox(label="Index(es) of quasi-unbiased estimator(s) with penalty(ies)", interactive=True,
                                            value="",
                                            info="Choose from 0 to length(Model)-1 (Example: [0] <=> Quasi-unbiased estimation with penalty for b[])")
                    pen_weight_text = gr.Textbox(label="Penalty(ies) of quasi-unbiased estimator(s) with penalty(ies)", interactive=True,
                                            value="",
                                            info="Set penalty(ies) in the range of 0.0 to 1.0 (Example: [0.1])")
                    ineq_index_text = gr.Textbox(label="Index(es) of quasi-unbiased estimator(s) with tolerance inequality(ies)", interactive=True,
                                            value="",
                                            info="Choose from 0 to length(Model)-1 (Example: [5,6,7] <=> Quasi-unbiased estimation with tolerance inequalities for b[0,1],b[0,2],b[0,3])")
                    ineq_tol_text = gr.Textbox(label="Tolerance(s) of quasi-unbiased estimator(s) with tolerance inequality(ies)", interactive=True,
                                            value="",
                                            info="Set tolerance(s) in the range of 0.0 to 1.0 (Example: [0.1, 0.1, 0.1])")
                    gr.Markdown(value="Settings of weight(s) (Note: If the input field is left blank, it(they) will be set automatically)")
                    kappa_weight_text = gr.Textbox(label="Weight(s) for variance(s) of estimator(s)", interactive=True,
                                            value="",
                                            info="Set non-negative weight(s) for variance(s) of estimator(s)  (Example: [1., 1., 1., 1., 1., 1., 1.])")
                    lambda_weight_text = gr.Textbox(label="Weight(s) for L1 regularization for all enumerated trial(s)", interactive=True,
                                            value="",
                                            info="Set weight(s) for L1 regularization for all enumerated trial(s) (Example: [7.0, 30.7, 30.3, 0.7, 26.5, 3.6, 6.6, 34.1, 2.8, 10.9, 17.4, 0.3, 16.6, 7.9, 7.4, 19.0])")
                start_button = gr.Button(value="Start")
            with gr.Group():
                export_all_information_csv_button = gr.Button(value="Export", visible=True)
                all_information_csv_file = gr.File(interactive=False, visible=False)
        with gr.Column():
            with gr.Group():
                output_text = gr.Textbox(label="Output",
                                    value="8 trial(s) [0, 3, 5, 6, 8, 11, 13, 14] were selected from a total of 16 trial(s).",
                                    interactive=False)
                output_dataframe = gr.Dataframe(label="Design matrix",
                                                show_label=True,
                                                value=GradioParameters.output_df.transpose().reset_index(),
                                                height=480)
                export_design_mat_csv_button = gr.Button("Export \"Design matrix\"")
                design_mat_csv_file = gr.File(interactive=False, visible=False)
            with gr.Group():
                all_patterns_button = gr.Button(value="Show \"All enumerated trial(s)\"")
                all_patterns_dataframe = gr.Dataframe(label="All enumerated trial(s)",
                                                show_label=True,
                                                value=GradioParameters.all_df.transpose().reset_index(),
                                                height=480,
                                                visible=False)
                export_all_patterns_csv_button = gr.Button(value="Export \"All enumerated trial(s)\"",
                                                           visible=False)
                all_patterns_csv_file = gr.File(interactive=False, visible=False)
    export_design_mat_csv_button.click(fn=export_design_mat_csv, inputs=[], outputs=[design_mat_csv_file])
    start_button.click(fn=start_explasso,
                        inputs=[levels_text, model_text,
                                eq_index_text,
                                lambda_scale_slider, lambda_randomness_slider,
                                pen_index_text, pen_weight_text,
                                ineq_index_text, ineq_tol_text,
                                kappa_weight_text, lambda_weight_text],
                        outputs=[output_text, output_dataframe, design_mat_csv_file,
                                 all_patterns_button, all_patterns_dataframe,
                                 export_all_patterns_csv_button, all_patterns_csv_file,
                                 all_information_csv_file])
    all_patterns_button.click(fn=show_all_pattern,
                              inputs=[all_patterns_button],
                              outputs=[all_patterns_button, all_patterns_dataframe,
                                       export_all_patterns_csv_button, all_patterns_csv_file])
    export_all_patterns_csv_button.click(fn=export_all_patterns_csv,
                                         inputs=[],
                                         outputs=[all_patterns_csv_file])
    export_all_information_csv_button.click(fn=export_all_information_csv,
                                         inputs=[],
                                         outputs=[all_information_csv_file])
demo.css = "footer {visibility: hidden}"
demo.launch(inbrowser=True, show_api=False, favicon_path="favicon.ico")
