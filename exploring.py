import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import pdb
import pickle


def list_top_ing(s):
# create dictionary of most popular 1000 ingredients

    slist = [y.strip() for x in s for y in x]

    top_items = sorted(slist, key=Counter(slist).get, reverse=True)

    top_items_unique = list(OrderedDict.fromkeys(top_items))

    top_items_unique.remove('')

    return top_items_unique[0:1000]


def clean_ing(s):
    # clean ingredients_text column
    s = s.str.replace('ingredients','')
    s = s.str.replace('ingredient','')
    s = s.str.replace('including','')
    s = s.str.replace('=','')
    s = s.str.replace('<','')
    s = s.str.replace('contains','')
    s = s.str.replace('organic','')
    s = s.str.replace('or less of','')
    s = s.str.replace('or less','')
    s = s.str.replace('less than of','')
    s = s.str.replace('less than','')
    s = s.str.replace('each of the following','')
    s = s.str.replace('the following','')
    s = s.str.replace('\+','')
    s = s.str.replace('/','')
    s = s.str.replace('@','')
    s = s.str.replace('as a ','')
    s = s.str.replace('contain','')
    s = s.str.replace('one or more of','')
    s = s.str.replace(r'[0-9]','')
    s = s.str.replace('>','')
    s = s.str.replace('to preserve freshness','')
    s = s.str.replace('%','')
    s = s.str.replace('&','')
    s = s.str.replace('#','')
    s = s.str.replace('\)','')
    s = s.str.replace('\]','')
    s = s.str.replace('\$','')
    s = s.str.replace(r'\*','')
    s = s.str.replace('"','')

    s = s.str.replace('\(', ',')
    s = s.str.replace('\[', ',')
    s = s.str.replace(':', ',')
    s = s.str.replace('\. ', ',')
    s = s.str.replace(';', ',')

    s = s.str.replace('\.', '')

    s = s.str.replace(' ,', ',')
    s = s.str.replace(', ', ',')

    s = s.str.strip()
    s = s.replace('\s+', ' ', regex=True)

    return s



def main():

    # read all the data and keep specific columns
    cols = ['product_name','ingredients_text','nutrition_grade_fr',
            'nutrition-score-fr_100g','sugars_100g', 'countries']

    all_data = pd.read_csv("en.openfoodfacts.org.products.tsv", sep='\t',
                           low_memory=False, usecols=cols, skipinitialspace=True)


    # keep only english ingredients and ingredients with score
    eng_data = all_data[(all_data['countries'] == 'US') &
                            (all_data['ingredients_text'].notnull()) &
                            (all_data['nutrition-score-fr_100g'].notnull())].reset_index()


    # preprocessing on ingredients: removing unwanted words and special characters
    eng_data.ingredients_text = eng_data.ingredients_text.str.lower()
    eng_data.ingredients_text = clean_ing(eng_data.ingredients_text)
    eng_data.ingredients_text = eng_data.ingredients_text.str.split(',')

    # list out the most popular ingredient
    top_ingredients = list_top_ing(eng_data.ingredients_text)

    # create a new column containing only ingredients that are part of the most popular
    eng_data['new_ing'] = np.nan
    eng_data['new_ing'] = eng_data['new_ing'].astype('object')

    for ind, ing in enumerate(eng_data.ingredients_text):
        new_list_ing = [x.strip() for x in ing if x in top_ingredients]
        eng_data.at[ind, 'new_ing'] = new_list_ing

    eng_data = eng_data[eng_data.new_ing.apply(len) > 0].reset_index()

    # store dataframe in a file to use for the neural network
    f = open('store.pckl', 'wb')
    pickle.dump(eng_data, f)
    f.close()


if __name__ == '__main__':
    main()
