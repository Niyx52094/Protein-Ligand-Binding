import numpy as np

def cal_NDCG(imitation_result):
    """
    the function to calculate normalized discounted cumulative gain,
    the ranking performance of the target ligand in the specific protein
    :param imitation_result: the dataset contains valid data with the shaple of positive (600), negative(493800)
    :return: the evaluation number
    """
#     length=len(imitation_result)
    ndcg=[]
    for k,v in imitation_result.items():
        length=len(v)
        idndcg=(2**1-1)/np.log2(2)
        for i in range(1,length):
            idndcg+=(2**0-1)/np.log2(i+2)

        true_ndcg=0
        for i in range(length):
            if k==v[i]:
                true_ndcg+=(2**1-1)/np.log2(i+2)
            else:
                true_ndcg+=(2**0-1)/np.log2(i+2)
        ndcg.append(true_ndcg/idndcg)
    return np.mean(ndcg)

def cal_sr(imitation_result):
    '''
    calcuate the success rate, the pecentage that the target ligand exist in the top 10 list.

    '''
    sr=0
    length=len(imitation_result)
    for k,v in imitation_result.items():
        if k in v:
            sr+=1
    return sr/length