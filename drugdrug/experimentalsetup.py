import numpy as np
import numpy.linalg
import time
import copy
import sklearn.metrics
from timeit import default_timer as timer
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import os
from sklearn.model_selection import KFold
import random
from numpy import NAN
from threestep import KroneckerKernel, NstepRegressor 

def read_tensor(filename, size_tensor):
	T = np.full(size_tensor, False)
	f =  open(filename, 'r')
	Lines = f.readlines()
	for line in Lines:
		sp = line.split()
		T[int(sp[0]),int(sp[1]),int(sp[2])] = True #sp[3]		
	return T

# multiplying with this tensor cancels the labels of the test triples.
def read_testfilter(filename, size_tensor):
    T = np.ones(size_tensor)
    f =  open(filename, 'r')
    Lines = f.readlines()
    if len(Lines[0].split())==3:
        print("testfilter triple")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),int(sp[2])] = np.nan #sp[3]		
            T[int(sp[1]),int(sp[0]),int(sp[2])] = np.nan
    if len(Lines[0].split())==2:
        print("testfilter pair")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),:] = np.nan #sp[3]         
            T[int(sp[1]),int(sp[0]),:] = np.nan

    return T

# multiplying with this data cancels the training triples and keeps test data.
def read_trainfilter(filename, size_tensor):
    T = np.full(size_tensor, np.nan)
    f =  open(filename, 'r')
    Lines = f.readlines()
    if len(Lines[0].split())==3:
        print("testfilter triple")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),int(sp[2])] = 1 #sp[3]             
            T[int(sp[1]),int(sp[0]),int(sp[2])] = 1
    if len(Lines[0].split())==2:
        print("testfilter pair")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),:] = 1 #sp[3]             
            T[int(sp[1]),int(sp[0]),:] = 1
    return T

# multiplying with this data sets test triple labels to zero.
def read_removetestlabels(filename,size_tensor):
    T = np.ones(size_tensor)
    f =  open(filename, 'r')
    Lines = f.readlines()
    if len(Lines[0].split())==3:
        print("testfilter triple")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),int(sp[2])] = 0 #sp[3]
            T[int(sp[1]),int(sp[0]),int(sp[2])] = 0
    if len(Lines[0].split())==2:
        print("testfilter pair")
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),:] = 0 #sp[3]
            T[int(sp[1]),int(sp[0]),:] = 0
    return T

# multiplying with this data cancels the training triples and keeps test data.
def read_removetrainlabels(filename, size_tensor):
    T = np.zeros(size_tensor)
    f =  open(filename, 'r')
    Lines = f.readlines()
    if len(Lines[0].split())==3:
        print('testfiler triple')
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),int(sp[2])] = 1 #sp[3]             
            T[int(sp[1]),int(sp[0]),int(sp[2])] = 1
    if len(Lines[0].split())==2:
        print('testfiler pair')
        for line in Lines:
            sp = line.split()
            T[int(sp[0]),int(sp[1]),:] = 1 #sp[3]             
            T[int(sp[1]),int(sp[0]),:] = 1
    return T



# Methods for calculating the evaluation given the label and score tensor

def AUROC(sc,la):
	try:
		la = la[~np.isnan(sc)]
		sc = sc[~np.isnan(sc)]
		return sklearn.metrics.roc_auc_score(la,sc)
	except:
		return np.nan

def AUPRC(sc, la):
	try:
		la = la[~np.isnan(sc)]
		sc = sc[~np.isnan(sc)]
		precision, recall, th = sklearn.metrics.precision_recall_curve(la,sc)
		return sklearn.metrics.auc(recall, precision)
	except:
		return np.nan
		
def AUPRC_ratio50(sc, la):
	try:
		la = la[~np.isnan(sc)]
		sc = sc[~np.isnan(sc)]
		precision, recall, th = sklearn.metrics.precision_recall_curve(la,sc)
		n1= len(la) - np.sum(la)  	#original number of negatives
		n2 = np.sum(la)*50			#wanted number of negatives to obtain steady ratio of 50
		precision = np.asarray(precision)
		precision = precision / (precision + n2/n1*(1-precision))
		return sklearn.metrics.auc(recall, precision)
	except:
		return np.nan

def AUPRC_ratio1(sc, la):
        try:
                la = la[~np.isnan(sc)]
                sc = sc[~np.isnan(sc)]
                precision, recall, th = sklearn.metrics.precision_recall_curve(la,sc)
                n1= len(la) - np.sum(la)        #original number of negatives
                n2 = np.sum(la)*1                      #wanted number of negatives to obtain steady ratio of 1
                precision = np.asarray(precision)
                precision = precision / (precision + n2/n1*(1-precision))
                return sklearn.metrics.auc(recall, precision)
        except:
                return np.nan


def evaluation(scores, labels, scorefunction, setting):
    num_cores = multiprocessing.cpu_count()
    
    if scorefunction == AUPRC:
        scorefunction = AUPRC_ratio50
    if setting==(False,False,False):
        sc = scores.flatten()
        la = labels.flatten()
        results.append(scorefunction(sc,la))
    if setting==(False,False,True): # edd: conditional on side-effect
        print('evaluaton', scores.shape, labels.shape)
        resultsparallel = Parallel(n_jobs=num_cores)(delayed(scorefunction)(scores[:,:,i].flatten(), labels[:,:,i].flatten()) for i in range(len(scores[1,1,:])))
        return resultsparallel		
    if setting==(True,True,False): # es: conditional on the first two drugs
        resultsparallel = Parallel(n_jobs=num_cores)(delayed(scorefunction)(scores[i,j,:], labels[i,j,:]) for i in range(len(scores[:,1,1])) for j in range(len(scores[1,:,1])))
        return resultsparallel	


def balanced_edd_evaluation(scores, labels, scorefunction):
    num_cores = multiprocessing.cpu_count()
    #resultsparallel = Parallel(n_jobs=num_cores)(delayed(balanced_score)(scores[:,:,i], labels[:,:,i], scorefunction) for i in range(len(scores[1,1,:])))            
    #return resultsparallel		
    if scorefunction == AUPRC:
        scorefunction = AUPRC_ratio1
    resultsparallel = Parallel(n_jobs=num_cores)(delayed(scorefunction)(scores[:,:,i].flatten(), labels[:,:,i].flatten()) for i in range(len(scores[1,1,:])))
    return resultsparallel


def balanced_score(sc, la, scorefunction):
    testddpairs = np.asarray(np.where(~np.isnan(sc))).transpose()
    positivelabels = []
    negativelabels = []
    positivescores = []
    negativescores = []
    for testddpair in testddpairs:
        if la[testddpair[0], testddpair[1]]==1:
            positivelabels.append(la[testddpair[0], testddpair[1]])
            positivescores.append(sc[testddpair[0], testddpair[1]])
        else:
            if len(negativelabels) < len(positivelabels):
                negativelabels.append(la[testddpair[0], testddpair[1]])
                negativescores.append(sc[testddpair[0], testddpair[1]])
    return scorefunction(np.asarray(positivescores+negativescores), np.asarray(positivelabels+negativelabels))


## General helpfunction for returning the testdrugs interval
def testdrugs_(i,n_drugs, n_splits):
    dx = int(n_drugs/n_splits+0.5)
    a,b = i*dx , min((i+1)*dx, n_drugs)
    return a,b


## Methods for preprocessing: create and save the kernels

#### Setting A = (True, True, True)
def K_effects_A(Y,dirname,n_splits):
    #define two non-overlapping section within one symmetric version of the tensor
    triple_a = np.asarray([[i,j,k] for i in range(len(Y)) for j in range(i+1,len(Y)-i) for k in range(Y.shape[2])])
    triple_b = np.asarray([[i,j,k] for i in range(len(Y)) for j in range(max(i+1,len(Y)-i),len(Y)) for k in range(Y.shape[2])])

    # first five folds in section a (section b is used for kernel construction), last five folds vice versa: ten folds in total. For each fold, the test indices are saved.
    n_splits=5
    kf = KFold(n_splits=n_splits, shuffle=True)
    i=0
    triples = triple_a
    for train_index, test_index in kf.split(triples): # save train and test ddpairs
        train, test = triples[train_index], triples[test_index]
        np.savetxt(dirname+"/test_"+str(i), np.asarray(test), fmt='%i')
        i+=1
    triples = triple_b
    for train_index, test_index in kf.split(triples): # save train and test ddpairs
        train, test = triples[train_index], triples[test_index]
        np.savetxt(dirname+"/test_"+str(i), np.asarray(test), fmt='%i')
        i+=1

    # for the first 5 folds, kernels are made based on the last ones. And vice versa
    pairs_a = np.asarray([[i,j] for i in range(len(Y)) for j in range(i+1,len(Y)-i)])
    pairs_b = np.asarray([[i,j] for i in range(len(Y)) for j in range(max(i+1,len(Y)-i),len(Y))])
    pairs = pairs_a
    effectfeatures = np.asarray([[Y[:,:,s][pairs[i,0], pairs[i,1]] for i in range(len(pairs))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8)
    Keffects_cos = cosine_similarity(effectfeatures, effectfeatures)
    for i in range(5):
        np.savetxt(dirname+"/Keffects_cos_"+str(i), Keffects_cos)
    pairs = pairs_b
    effectfeatures = np.asarray([[Y[:,:,s][pairs[i,0], pairs[i,1]] for i in range(len(pairs))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8)
    Keffects_cos = cosine_similarity(effectfeatures, effectfeatures)
    for i in range(5,10):
        np.savetxt(dirname+"/Keffects_cos_"+str(i), Keffects_cos)

def K_drugs_A(dirname, l):
	mono = pd.read_csv('Data_Decagon/bio-decagon-mono/bio-decagon-mono.csv')
	drugsinds = np.loadtxt("Data_Indexed/drugsinds", str)
	drugsmap = dict(zip(drugsinds[:,0], drugsinds[:,1].astype(int)))
	indivsideeffectmap = dict(zip(np.unique(mono['Individual Side Effect']), range(len(np.unique(mono['Individual Side Effect'])))))
	m = np.zeros((len(drugsmap), len(indivsideeffectmap)))
	for i in range(len(mono)):
		d = drugsmap[mono.iloc[i,0]]
		se = indivsideeffectmap[mono.iloc[i,1]]
		m[d,se] = 1
	Kdrugs_rbf = rbf_kernel(m, m)
	np.savetxt(dirname +"/Kdrugs_rbf", Kdrugs_rbf[:l,:l])

def hdv_drugs_A(dirname):
    try:
        hdv_drugs = np.loadtxt(dirname+"/hdv_drugs")
    except:
        mono = pd.read_csv('Data_Decagon/bio-decagon-mono/bio-decagon-mono.csv')
        drugsinds = np.loadtxt("Data_Indexed/drugsinds", str)
        drugsmap = dict(zip(drugsinds[:,0], drugsinds[:,1].astype(int)))
        indivsideeffectmap = dict(zip(np.unique(mono['Individual Side Effect']), range(len(np.unique(mono['Individual Side Effect'])))))
        m = np.zeros((len(drugsmap), len(indivsideeffectmap)))
        for i in range(len(mono)):
            d = drugsmap[mono.iloc[i,0]]
            se = indivsideeffectmap[mono.iloc[i,1]]
            m[d,se] = 1
        P = np.random.choice([-1,1], (10184,10000))
        hdv_drugs = np.sign(np.matmul(m, P))
        #hdv_drugs = np.matmul(m, P)
        np.savetxt(dirname+"/hdv_drugs", hdv_drugs)
    return hdv_drugs

def hdv_effects_A(Y, dirname):
    try:
        hdv_effects_a, hdv_effects_b = np.loadtxt(dirname+"/hdv_effects_a"), np.loadtxt(dirname+"/hdv_effects_b")
    except:
    
        # a is for the first five folds and b for the last five folds, this is determined in de splitting in K_effects_a
        pairs_a = np.asarray([[i,j] for i in range(len(Y)) for j in range(i+1,len(Y)-i)])
        pairs_b = np.asarray([[i,j] for i in range(len(Y)) for j in range(max(i+1,len(Y)-i),len(Y))])

        effectfeatures_a = np.asarray([[Y[:,:,s][pairs_a[i,0], pairs_a[i,1]] for i in range(len(pairs_a))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8)
        effectfeatures_b = np.asarray([[Y[:,:,s][pairs_b[i,0], pairs_b[i,1]] for i in range(len(pairs_b))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8)

        hdv_effects_a = np.array([np.matmul(effectfeatures_a, np.random.choice([-1,1], effectfeatures_a.shape[1])) for i in range(10000)])
        np.savetxt(dirname+"/hdv_effects_a", hdv_effects_a)
        
        hdv_effects_b = np.array([np.matmul(effectfeatures_b, np.random.choice([-1,1], effectfeatures_b.shape[1])) for i in range(10000)])    
        np.savetxt(dirname+"/hdv_effects_b", hdv_effects_b)
    return np.sign(hdv_effects_a.transpose()), np.sign(hdv_effects_b.transpose())
    #return hdv_effects_a.transpose(), hdv_effects_b.transpose()
    	
#### Setting T1 = (...)
def K_effects_T1(Y,dirname,n_splits):
    #define the drug-drug pairs.
    pairs = np.asarray([[i,j] for i in range(len(Y)) for j in range(i+1,len(Y))])
    # split them into folds, add the indexes of side effects and write triples to the files
    n_splits=10
    kf = KFold(n_splits=n_splits, shuffle=True)
    i=0
    for train_index, test_index in kf.split(pairs): # save train and test ddpairs
        # save the test pairs
        train, test = pairs[train_index], pairs[test_index]
        np.savetxt(dirname+"/test_"+str(i), np.asarray(test), fmt='%i')
        
        # use the train pairs to construct kernels
        effectfeatures = np.asarray([[Y[:,:,s][train[i][0], train[i][1]] for i in range(len(train))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8)
        Keffects_cos = cosine_similarity(effectfeatures, effectfeatures)
        np.savetxt(dirname+"/Keffects_cos_"+str(i), Keffects_cos)

        i+=1

def hdv_effects_T1_help(Y, dirname, n_splits):
    effectfeatures_=[]
    for split in range(n_splits):
        beginexp = timer()
        pairs = np.loadtxt(dirname+"/test_"+str(split)).astype(int)
        effectfeatures_.append( np.asarray([[Y[:,:,s][pairs[i,0], pairs[i,1]] for i in range(len(pairs))] for s in range(len(Y[1,1,:]))]).astype(dtype=np.uint8))
        
        print(timer()-beginexp)

    D=10000
    hdv_effects_ = [np.zeros((effectfeatures_[0].shape[0],D)) for split in range(n_splits)]
    for i in range(D):
        #Pi = np.random.choice([-1,1], effectfeatures_[0].shape[1])
        for split in range(n_splits):
            hdv_effects_[split][:,i] = np.matmul(effectfeatures_[split], np.random.choice([-1,1], effectfeatures_[0].shape[1])) 

    for split in range(n_splits):
        np.savetxt(dirname+"/hdv_effects_help_"+str(split), hdv_effects_[split])
    
def hdv_effects_T1(Y,dirname, n_splits):
    try:
        return np.asarray([np.loadtxt(dirname+"/hdv_effects_help_"+str(split)) for split in range(n_splits)])
    except:
        hdv_effects_T1_help(Y, dirname, n_splits)
        return np.asarray([np.loadtxt(dirname+"/hdv_effects_help_"+str(split)) for split in range(n_splits)])
def K_drugs_T1(dirname, l):
    return K_drugs_A(dirname,l)


#### Setting B = (True, True, True)

def K_effects_B(Y, dirname, n_splits):
	for i in range(n_splits):
		testdrugs = np.arange(testdrugs_(i,len(Y),n_splits)[0],testdrugs_(i,len(Y),n_splits)[1])
		Ytrain = np.delete(np.delete(Y, testdrugs, axis=1), testdrugs, axis = 0)
		effectfeatures = Ytrain.reshape(Ytrain.shape[0]*Ytrain.shape[1], Ytrain.shape[2]).transpose()
		Keffects_cos = cosine_similarity(effectfeatures, effectfeatures)
		np.savetxt(dirname+ "/Keffects_cos_"+str(i), Keffects_cos)
		del testdrugs
		del Ytrain
		del effectfeatures
		del Keffects_cos
def K_drugs_B(dirname, l):
	K_drugs_A(dirname, l)
    
def hdv_effects_B_help(Y, dirname, n_splits):
    effectfeatures_=[[] for i in range(n_splits)]
    for split1 in range(n_splits):
        testdrugs1 = np.arange(testdrugs_(split1,len(Y), n_splits)[0],testdrugs_(split1,len(Y),n_splits)[1])
        for split2 in range(split1): #symmetry
            testdrugs2 = np.arange(testdrugs_(split2,len(Y), n_splits)[0],testdrugs_(split2,len(Y),n_splits)[1])
            effectfeatures_[split1].append((Y[testdrugs1,:,:][:,testdrugs2,:]).reshape((len(testdrugs1)*len(testdrugs2),Y.shape[2])).transpose())
        for split2 in range(split1, n_splits): #symmetry
            testdrugs2 = np.arange(testdrugs_(split2,len(Y), n_splits)[0],testdrugs_(split2,len(Y),n_splits)[1])
            effectfeatures_[split1].append((Y[testdrugs2,:,:][:,testdrugs1,:]).reshape((len(testdrugs1)*len(testdrugs2),Y.shape[2])).transpose())
    
    D=10000
    hdv_effects_ = [[np.zeros((effectfeatures_[0][0].shape[0],D)) for split in range(n_splits)]for split in range(n_splits)]
    beginexp=timer()
    for i in range(D):
        for split1 in range(n_splits):
            for split2 in range(split1, n_splits):
                hdv_effects_[split1][split2][:,i] = hdv_effects_[split2][split1][:,i] = np.matmul(effectfeatures_[split1][split2], np.random.choice([-1,1], effectfeatures_[split1][split2].shape[1]))
            #hdv_effects_[split][:,i] = np.matmul(effectfeatures_[split], np.random.choice([-1,1], effectfeatures_[split].shape[1])) 
    print(timer()-beginexp)

    for split1 in range(n_splits):
        for split2 in range(n_splits):
            np.savetxt(dirname+"/hdv_effects_help_"+str(split1)+"_"+str(split2), hdv_effects_[split1][split2])
    
def hdv_effects_B(Y,dirname, n_splits):
    try:
        return np.asarray([[np.loadtxt(dirname+"/hdv_effects_help_"+str(split1)+"_"+str(split2)).astype('int16') for split1 in range(n_splits)] for split2 in range(n_splits)])
    except:
        hdv_effects_B_help(Y, dirname, n_splits)
        return np.asarray([[np.loadtxt(dirname+"/hdv_effects_help_"+str(split1)+"_"+str(split2)).astype('int16') for split1 in range(n_splits)] for split2 in range(n_splits)])


#### Setting C = (True, True, True)

def K_effects_C(Y, dirname, n_splits):
	K_effects_B(Y, dirname, n_splits)
	
def K_drugs_C(dirname, l):
	K_drugs_A(dirname, l)



## Methods for determining the validation performances on the hyperparameter grid.

def evaluate_grid(hypamsets, setting, Y, canceltrain, canceltest, testfilter, K, experimentnumber, dirname):
    f = dirname+'/hypamoptimization_'+str(experimentnumber)
    if os.path.isfile(f) == False:
        print('true')
        with open(dirname+'/hypamoptimization_'+str(experimentnumber), 'a') as the_file:
            the_file.write("\t".join(['h1','h2','h3','(False, False, True)AUROC','(True, True, False)AUROC', '(False, False, True)AUPRC','(True, True, False)AUPRC']) + "\n")
        the_file.close()
        print('the file initialized')		
    for hypamset in hypamsets:
        print(hypamset)
        nstep = NstepRegressor(hypamset)
        start = timer()
        if setting==(True, True, True) or setting=='T1':
            print('tensorcompletion and imputation', setting)
            Yest = nstep.fit_predict_LO(K,Y*canceltest, setting)
            #Yest = Yest*testfilter
            Yimputed = Y + (Yest-Y)*canceltrain #for the test data, Y is replaced by an estimation
            Yest = nstep.fit_predict_LO(K,Yimputed,setting)
            Yest = Yest*testfilter
        else:
            Yest = nstep.fit_predict_LO(K,Y, setting)
        end = timer()
        estimatationt = end-start
        start = timer()
        edd_AUROC = np.nanmean(evaluation(Yest,Y,AUROC,(False,False,True)))
        es_AUROC  = np.nanmean(evaluation(Yest,Y,AUROC,(True, True, False)))
        edd_AUPRC = np.nanmean(evaluation(Yest,Y,AUPRC,(False,False,True)))
        es_AUPRC  = np.nanmean(evaluation(Yest,Y,AUPRC,(True, True, False)))  
        end = timer()
        evalutatingtime = end-start
        print('est: '+  str(estimatationt) + '     eval: '+ str(evalutatingtime))           
        del Yest
        with open(dirname+'/hypamoptimization_'+str(experimentnumber), 'a') as the_file:
            the_file.write("\t".join(np.asarray(hypamset + [edd_AUROC,es_AUROC, edd_AUPRC,es_AUPRC]).astype(str)) + "\n")
        the_file.close()
        del edd_AUROC
        del es_AUROC
        del edd_AUPRC
        del es_AUPRC
	
#### Setting A = (True, True, True)

def evaluate_grid_A(Y, hypamsets, dirname, experimentnumber, canceltrain, canceltest, testfilter):
    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
    K = KroneckerKernel([K1, K2, K3])
    evaluate_grid(hypamsets, (True, True, True), Y, canceltrain, canceltest, testfilter, K, experimentnumber, dirname) #here the complete Y tensor is given and remains bool tensor (for fitting in memory). Test values in the estimated tensor will be set to nan to remove them from evaluation.


#### Setting T1
def evaluate_grid_T1(Y, hypamsets, dirname, experimentnumber,canceltrain, canceltest, testfilter):
    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
    K = KroneckerKernel([K1, K2, K3])
    evaluate_grid(hypamsets, 'T1', Y, canceltrain, canceltest, testfilter, K, experimentnumber, dirname) #here the complete Y tensor is given and remains bool tensor (for fitting in memory). Test values in the estimated tensor will be set to nan to remove them from evaluation.



#### Setting B = (False, True, True)

def evaluate_grid_B(Y, hypamsets, dirname, n_splits, experimentnumber):
    testdrugs = np.arange(testdrugs_(experimentnumber,len(Y), n_splits)[0],testdrugs_(experimentnumber,len(Y),n_splits)[1])
    Ytrain = np.delete(np.delete(Y, testdrugs, axis=0), testdrugs, axis=1).astype('float64')
    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
    K1 = K2 = np.delete(np.delete(K1, testdrugs, axis=0), testdrugs,axis=1)
    K = KroneckerKernel([K1, K2, K3])
    evaluate_grid(hypamsets, (False, True, True), Ytrain.astype('float64'), None, None, None, K, experimentnumber, dirname)

#### Setting C =  (False, False, True)

def evaluate_grid_C(Y, hypamsets, dirname, n_splits, experimentnumber):
    testdrugs = np.arange(testdrugs_(experimentnumber,len(Y), n_splits)[0],testdrugs_(experimentnumber,len(Y),n_splits)[1])
    Ytrain = np.delete(np.delete(Y, testdrugs, axis=0), testdrugs, axis=1)
    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
    K1 = K2 = np.delete(np.delete(K1, testdrugs, axis=0), testdrugs,axis=1)
    K = KroneckerKernel([K1, K2, K3])
    evaluate_grid(hypamsets, (False, False, True), Ytrain, None, None, None, K, experimentnumber, dirname)


## Methods for determining performances on independent test set at the optimum hyperparameter of the validationset


def final_pooling_evaluation(Y,setting,dirname,n_splits,metrics,schemes):
    for metric in metrics:
        for scheme in schemes:
            Yhat = np.zeros(Y.shape)
            if setting == (False,True,True) or setting == (False,False,True): # in these settings there is not in every fold an estimation, so we cannot simply add predictions, non predicted must remain nan
                Yhat[:] = np.nan
            for experimentnumber in range(n_splits):
                opi = str(scheme)+metric.__name__
                optimizationresults = pd.read_csv(dirname+'/hypamoptimization_'+str(experimentnumber), sep='\t')#, names=colnames)
                o = list(optimizationresults[optimizationresults[opi] == max(optimizationresults[opi])].iloc[0,0:3])
                nstep = NstepRegressor(o)
                if setting == (True, True, True) or setting == 'T1':

                    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
                    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
                    K = KroneckerKernel([K1, K2, K3])

                    #load the matrices into Ktrain kroneckerkernel
                    canceltest = read_removetestlabels(dirname+'/test_'+str(experimentnumber),Y.shape)
                    canceltrain = read_removetrainlabels(dirname+'/test_'+str(experimentnumber),Y.shape)
                    Yest = nstep.fit_predict_LO(K,Y*canceltest, setting)
                    Yimputed = Y + (Yest-Y)*canceltrain #for the test data, Y is replaced by an estimation. This improves quality of training data
                    Yest = nstep.fit_predict_LO(K,Yimputed,setting)
                    Yhat = Yhat + Yest*canceltrain # put training estimations to zero and add these to the Yhat, if this is done for every fold, the Yhat tensor is a pooled estimation from the folds

                if setting == (False, True, True):
                    testdrugs = np.arange(testdrugs_(experimentnumber,len(Y),n_splits)[0],testdrugs_(experimentnumber,len(Y), n_splits)[1])
                    Ytrain = np.delete(np.delete(Y, testdrugs, axis=0), testdrugs, axis=1)
                    Ytest = np.delete(Y, testdrugs, axis=1)[testdrugs[0]:testdrugs[-1], :,:]
                    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
                    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
                    Ktrain = KroneckerKernel([np.delete(np.delete(K1, testdrugs, axis=0), testdrugs,axis=1), np.delete(np.delete(K2, testdrugs, axis=0), testdrugs,axis=1), K3])
                    Ktest = KroneckerKernel([np.delete(K1, testdrugs, axis=1)[testdrugs[0]:testdrugs[-1], :], np.delete(np.delete(K2, testdrugs, axis=0), testdrugs,axis=1), K3])
                    nstep.fit(Ktrain, Ytrain.astype('float64'))
                    Yest = nstep.predict(Ktest) #.astype('float32')
                    # assign the predictions correctly: note we did not, due to symmetry estimate for a diagonal block (these acutally coresspond to setting C = (False,False,True)). This is why there is a upper part assignment and lower part assignment (below the diagonal block)
                    print(Yhat.shape, Ytrain.shape, Ytest.shape, Yest.shape,len(testdrugs))

                    Yhat[testdrugs[0]:testdrugs[-1], :testdrugs[0] , :] = Yest[:,:testdrugs[0],:]
                    Yhat[testdrugs[0]:testdrugs[-1], testdrugs[-1]+1: , :] = Yest[:,testdrugs[0]:,:]

                if setting == (False, False, True):
                    testdrugs = np.arange(testdrugs_(experimentnumber,len(Y),n_splits)[0],testdrugs_(experimentnumber,len(Y), n_splits)[1])
                    Ytrain = np.delete(np.delete(Y, testdrugs, axis=0), testdrugs, axis=1)
                    Ytest = Y[testdrugs[0]:testdrugs[-1], testdrugs[0]:testdrugs[-1],:]
                    K1 = K2 =  np.loadtxt(dirname+"/Kdrugs_rbf")
                    K3 = np.loadtxt(dirname+"/Keffects_cos_"+str(experimentnumber))
                    Ktrain = KroneckerKernel([np.delete(np.delete(K1, testdrugs, axis=0), testdrugs,axis=1), np.delete(np.delete(K2, testdrugs, axis=0), testdrugs,axis=1), K3])
                    Ktest = KroneckerKernel([np.delete(K1, testdrugs, axis=1)[testdrugs[0]:testdrugs[-1], :], np.delete(K2, testdrugs, axis=1)[testdrugs[0]:testdrugs[-1], :], K3])

                    nstep.fit(Ktrain, Ytrain.astype('float64'))
                    Yest = nstep.predict(Ktest) #.astype('float32')
                    # assign the predictions correctly: here we assign the diagonal blocks. Note that here we need to set the diagonal elements to nan to neglegt in the evaluation.
                    Yhat[testdrugs[0]:testdrugs[-1], testdrugs[0]:testdrugs[-1] , :] = Yest
                    ndiag = len(Yhat)
                    for i in range(ndiag):
                        Yhat[i,i,:] = np.nan


                
                print(experimentnumber)

            e = evaluation(Yhat, Y, metric, scheme)   #########################################################################################

            with open(dirname+'/test_pool' + opi, 'a') as the_file:
                the_file.write("\t".join(np.asarray(e).astype(str)))
                print("evaluated and saved to "+dirname+'/test_pool' + opi)
            the_file.close()


            if scheme == (False, False, True) and setting== (True,True,True):
                ebalanced = balanced_edd_evaluation(Yhat, Y, metric)
                with open(dirname+'/test_pool_balanced' + opi, 'a') as the_file:
                    print('balanced evaluation done')
                    the_file.write("\t".join(np.asarray(ebalanced).astype(str)) + "\n")
                the_file.close()

