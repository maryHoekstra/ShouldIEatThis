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
    top_items_unique.remove('a')
    top_items_unique.remove('b')
    top_items_unique.remove('c')
    top_items_unique.remove('e')
    top_items_unique.remove('l')
    top_items_unique.remove('s')
    top_items_unique.remove('w')
    top_items_unique.remove('as')
    top_items_unique.remove('as s')

    return top_items_unique[0:1000]


def clean_ing(s):
    # clean ingredients_text column

    s = s.str.replace('=','')
    s = s.str.replace('<','')
    s = s.str.replace('\+','')
    s = s.str.replace('/','')
    s = s.str.replace('@','')
    s = s.str.replace(r'[0-9]','')
    s = s.str.replace('>','')
    s = s.str.replace('%','')
    s = s.str.replace('&','')
    s = s.str.replace('#','')
    s = s.str.replace('\)','')
    s = s.str.replace('\]','')
    s = s.str.replace('\}','')
    s = s.str.replace('\$','')
    s = s.str.replace(r'\*','')
    s = s.str.replace('"','', regex = False)
    s = s.str.replace('?','', regex = False)
    s = s.str.replace('|','')
    s = s.str.replace('½ of','')
    s = s.str.replace('†','')
    s = s.str.replace('!','')
    s = s.str.replace('_','')

    s = s.str.replace('a preservative','')
    s = s.str.replace('added to enhance freshness','')
    s = s.str.replace('added to enhance tartness','')
    s = s.str.replace('added to help mantain firmness','')
    s = s.str.replace('added to help prevent caking','')
    s = s.str.replace('added to make free flowing','')
    s = s.str.replace('added to retard oxidation','')
    s = s.str.replace('added to retard rancidity','')
    s = s.str.replace('added to retared spoilage','')
    s = s.str.replace('an emulsifier','')
    s = s.str.replace('as a ','')
    s = s.str.replace('contains','')
    s = s.str.replace('contain','')
    s = s.str.replace('of each of the following','')
    s = s.str.replace('each of the following','')
    s = s.str.replace('each of','')
    s = s.str.replace('each','')
    s = s.str.replace('emulsifier','')
    s = s.str.replace('for filling','')
    s = s.str.replace('for firmness','')
    s = s.str.replace('for flavor and color','')
    s = s.str.replace('for flavoring only','')
    s = s.str.replace('for flavoring','')
    s = s.str.replace('for freshness and ascorbic acid','')
    s = s.str.replace('for freshness','')
    s = s.str.replace('for improved baking','')
    s = s.str.replace('for improved caking','')
    s = s.str.replace('for leavening','')
    s = s.str.replace('for life - social responsibility certified','')
    s = s.str.replace('for moisture retention','')
    s = s.str.replace('for moisture','')
    s = s.str.replace('for ph balance','')
    s = s.str.replace('for quality','')
    s = s.str.replace('ingredients','')
    s = s.str.replace('ingredient','')
    s = s.str.replace('including','')
    s = s.str.replace('includes','')
    s = s.str.replace('less than of','')
    s = s.str.replace('less than','')
    s = s.str.replace('one or more of','')
    s = s.str.replace('organic','')
    s = s.str.replace('or less of','')
    s = s.str.replace('less of','')
    s = s.str.replace('or less','')
    s = s.str.replace('preservative','')
    s = s.str.replace('the following','')
    s = s.str.replace('to color','')
    s = s.str.replace('to conservative','')
    s = s.str.replace('to conservat','')
    s = s.str.replace('to enhance color','')
    s = s.str.replace('to enhance flavor','')
    s = s.str.replace('to enhance texture','')
    s = s.str.replace('to enrich the milk','')
    s = s.str.replace('to ensure freshness','')
    s = s.str.replace('to freshness','')
    s = s.str.replace('to fry','')
    s = s.str.replace('to help flavor','')
    s = s.str.replace('to help maintain firmness','')
    s = s.str.replace('to help maintain freshness','')
    s = s.str.replace('to help preserve freshness','')
    s = s.str.replace('to help promote color retention','')
    s = s.str.replace('to help protect flavor not found in regular cottage cheese','')
    s = s.str.replace("to help protect flavor' not found in regular cottage cheese",'')
    s = s.str.replace('to help protect flavor','')
    s = s.str.replace('to help protect freshness','')
    s = s.str.replace('to help protect','')
    s = s.str.replace('to keep freshness','')
    s = s.str.replace('to maintain color','')
    s = s.str.replace('to maintain firmness','')
    s = s.str.replace('to maintain flavor and firmness','')
    s = s.str.replace('to maintain flavor and freshness','')
    s = s.str.replace('to maintain flavor','')
    s = s.str.replace('to maintain freshness','')
    s = s.str.replace('to maintain moisture','')
    s = s.str.replace('to maintain natural color','')
    s = s.str.replace('to maintain quality','')
    s = s.str.replace('to maintain texture','')
    s = s.str.replace('to obtain freshness','')
    s = s.str.replace('to preserve color','')
    s = s.str.replace('to preserve flavor and color','')
    s = s.str.replace('to preserve flavor','')
    s = s.str.replace('to preserve freshness','')
    s = s.str.replace('to preserve natural pepper color','')
    s = s.str.replace('to preserve natural potato color','')
    s = s.str.replace('to preserve quality','')
    s = s.str.replace('to preserve the natural wheat flavor','')
    s = s.str.replace('to preserve','')
    s = s.str.replace('to prevent browning','')
    s = s.str.replace('to prevent clumping enzymes','')
    s = s.str.replace('to prevent caking','')
    s = s.str.replace('to prevent clumping','')
    s = s.str.replace('to prevent foaming','')
    s = s.str.replace('to prevent freshness','')
    s = s.str.replace('to prevent separation','')
    s = s.str.replace('to prevent spoilage','')
    s = s.str.replace('to prevent sticking','')
    s = s.str.replace('to prevent the formation of struvite crystals','')
    s = s.str.replace('to prevent','')
    s = s.str.replace('to promote browning','')
    s = s.str.replace('to promote color retention','')
    s = s.str.replace('to promote color and flavor retention','')
    s = s.str.replace('to promote color','')
    s = s.str.replace('to promote freshness','')
    s = s.str.replace('to promote oven browning','')
    s = s.str.replace('to promote','')
    s = s.str.replace('to protect color and flavor','')
    s = s.str.replace('to protect color retention','')
    s = s.str.replace('to protect color','')
    s = s.str.replace('to protect flavor','')
    s = s.str.replace('to protect freshness','')
    s = s.str.replace('to protect from dehydration','')
    s = s.str.replace('to protect quality','')
    s = s.str.replace('to protect texture','')
    s = s.str.replace('to protect','')
    s = s.str.replace('to provide thickness','')
    s = s.str.replace('to retain freshness','')
    s = s.str.replace('to retain moisture','')
    s = s.str.replace('to reconstitute citric acid','')
    s = s.str.replace('to reconstitute','')
    s = s.str.replace('to reduce caking','')
    s = s.str.replace('to retain color and sodium benzoate','')
    s = s.str.replace('to retain color','')
    s = s.str.replace('to retain fish moisture','')
    s = s.str.replace('to retain flavor','')
    s = s.str.replace('to retain natural color','')
    s = s.str.replace('to retain natural juices','')
    s = s.str.replace('to retain natural juice','')
    s = s.str.replace('to retain natural moisture','')
    s = s.str.replace('to retain whiteness','')
    s = s.str.replace('to retard spoilage','')
    s = s.str.replace('to return freshness','')
    s = s.str.replace('to stabilize color','')
    s = s.str.replace('two percent','')
    s = s.str.replace('percent','')
    s = s.str.replace('used to protect quality','')
    s = s.str.replace("what's in it",'')
    s = s.str.replace('with not more than less of','')
    s = s.str.replace('with not more than of','')
    s = s.str.replace('with not more than','')
    s = s.str.replace('with','')
    s = s.str.replace("you'll love",'')

    s = s.str.replace('apples','apple')
    s = s.str.replace('artificial colors','artificial color')
    s = s.str.replace('artificial flavors','artificial flavor')
    s = s.str.replace('blueberries','blueberry')
    s = s.str.replace('green onions','green onion')
    s = s.str.replace('grapes','grape')
    s = s.str.replace('raspberries','raspberry')
    s = s.str.replace('strawberries','strawberry')

    s = s.str.replace('\(', ',')
    s = s.str.replace('\[', ',')
    s = s.str.replace('\{', ',')
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
