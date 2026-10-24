import numpy as np

# Hungarian Algorithm for matching labels of fit states to labels of true states
def Hungarian(cost_mat, mod_ind_hist):
    nS = cost_mat.shape[0] #Gamma_fit.shape[2]
    #nChan = int(Gamma_fit.shape[0]/2)
    #corr_mat = np.zeros((nS,nS))

    # Get correlations for each fit Gamma with each true Gamma
    # for i in range(nS):
    #    for j in range(nS):
    #        corr_mat[i,j] = np.corrcoef(Gamma_fit[:,:,i].flatten(),Gamma_true[:,:,j].flatten())[0,1] #\
                            #+ np.corrcoef(Gamma_fit[:nChan,:nChan,i].flatten(),Gamma_true[:nChan,:nChan,j].flatten())[0,1] \
                            #+ np.corrcoef(Gamma_fit[nChan:,:nChan,i].flatten(),Gamma_true[nChan:,:nChan,j].flatten())[0,1]

    # print(corr_mat)
    # Subtract the maximum from each row
    cost_mat = cost_mat - np.max(cost_mat,1)[:,None]
    unique_cols = len(np.unique(np.argwhere(cost_mat == 0)[:,1]))
    print(cost_mat)

    # If there is no zero-cost solution
    if unique_cols < nS:
        # Subtract the maximum from each column
        cost_mat = cost_mat - np.max(cost_mat,0)[None,:]
        unique_cols = len(np.unique(np.argwhere(cost_mat == 0)[:,1]))
        # If there is still no zero-cost solution
        if unique_cols < nS:
            print('need to implement steps 3-5')

    print(unique_cols)
    print(cost_mat)
    cost_mat_s2 = cost_mat
    # Assign labels
    labels = []
    stars = []
    primes = []
    iter=0
    while len(labels) < nS:
        for state in range(nS):
            n_zeros = len(np.argwhere(cost_mat[state,:] == 0))
            if n_zeros == 1:
                label = np.argwhere(cost_mat[state,:] == 0)[0,0]
                labels.append((state,label))
                cost_mat[:,label] = -2
            elif iter>0 and n_zeros > 1:
                print(len(np.argwhere(cost_mat[state,:] == 0)))
                star = np.argwhere(cost_mat[state,:] == 0)[0,0]
                stars.append((state,star))
                cost_mat[:,star] = -2
            print(cost_mat)
            print(len(labels))
        iter += 1

    # Re-label mod_ind_hist
    mod_ind_hist_new = np.zeros(mod_ind_hist.shape)
    for i in range(nS):
        label = labels[i]
        mod_ind_hist_new[mod_ind_hist == label[0]] = label[1]
    
    return labels, mod_ind_hist_new