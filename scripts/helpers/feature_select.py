# Tools for feature selection 

import numpy as np

def get_linked_groups(corrcoeff, thresh):
    """
    Get groups of directly and indirectly correlated columns
    
    Args:
        corrcoeff: correlation coefficient matrix
        thresh: correlation coefficient threshold for linking
    """
    correlated_columns = np.where(np.abs(np.triu(np.nan_to_num(corrcoeff,0),1))>=thresh)
    
#     corr_nodiag = corrcoeff - np.diag(np.diag(corrcoeff))
#     correlated_columns = np.where(np.abs(np.nan_to_num(corr_nodiag,0))>=thresh)
    
    groups = []
    for num in np.unique(correlated_columns[0]):
        in_set = [num in numset for numset in groups]
        if max(in_set,default=False)==False:
            #if not already in a set, get correlated var nums and check if they belong to an existing set
            cnums = correlated_columns[1][np.where(correlated_columns[0]==num)]
            numset = set([num] + list(cnums))
            #check if numset intersects an existing set
            intersect = [numset & group for group in groups]
            if len(intersect) > 0:
                intersect_group = intersect[np.argmax(intersect)]
            else:
                intersect_group = []
            #if intersects existing set, add to set
            if len(intersect_group) > 0:
                intersect_group |= numset
                #print('case 1 existing group:', num, intersect_group)
            #otherwise, make new set
            else:
                groups.append(numset)
                #print('new group:', num, cnums)
        else:
            #if already in a set, get correlated var nums and add to set
            group = groups[in_set.index(True)]
            cnums = correlated_columns[1][np.where(correlated_columns[0]==num)]
            group |= set(cnums) #union
            #print('case 2 existing group:', num, group)
    
    #some links may not be captured. Ex: 1 -> {4,5}. 2 -> 3. 3 -> 4. Now groups are: {1,4,5}, {2,3,4} - need to combine
    #safety net - combine groups that share common elements
    for i,group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            if len(group1 & group2) > 0:
                group1 |= group2
                groups.remove(group2)
    
    return groups
	
def choose_independent_features(corrcoeff,thresh,response_col=0,drop_invariant=True):
    """
    Choose features that correlate best with the response and are not correlated with each other.
    Identify correlation groups and keep the single feature with the strongest correlation to the response from each group.
    
    Args:
        corrcoeff: correlation coefficient matrix. Should include the response
        thresh: correlation coefficient threshold
        response_col: column index for response. Default 0
        drop_invariant: if True, drop columns with zero variance
    """
    groups = get_linked_groups(corrcoeff,thresh=thresh)
    #get list of all columns in linked groups
    correlated = sum([list(group) for group in groups],[])
    #get list of unlinked columns 
    independent = set(np.arange(corrcoeff.shape[0])) - set(correlated)
    #for each linked group, keep the feature that correlates most strongly with the response
    keep = []
    for group in groups:
        max_idx = np.argmax(np.abs(corrcoeff[response_col,list(group)]))
        keep.append(list(group)[max_idx])

    keep += list(independent)
    
    #check
    check1 = (len(correlated) + len(independent) == corrcoeff.shape[0])
    check2 = (len(correlated) + len(keep) - len(groups) == corrcoeff.shape[0])
    if min(check1,check2)==False:
        raise Exception('Number of correlated and independent features do not match total number')
    
    if drop_invariant==True:
        invariant = list(np.where(np.nan_to_num(fcorr,0)[0]==0)[0])
        keep = list(set(keep) - set(invariant))
    
    #always keep the response
    if response_col not in keep:
        keep.append(response_col)
    
    keep.sort()
    
    return keep