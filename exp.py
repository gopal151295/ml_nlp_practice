import numpy as np

mystr = "cat dog apple lion NYC love"
random = np.random.rand(6)

mystr_split = mystr.split()
mystr_split_zip = zip(mystr_split, random)
mystr_split_zip_list = list(mystr_split_zip)
mystr_split_zip_list_dict = dict(mystr_split_zip_list)
