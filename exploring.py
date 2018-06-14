import pandas as pd


def main():
    # read all the data and keep specific columns
    cols = ['product_name','ingredients_text','nutrition_grade_fr',
            'nutrition-score-fr_100g','sugars_100g', 'countries']

    all_data = pd.read_csv("en.openfoodfacts.org.products.tsv", sep = '\t',
                            low_memory=False, usecols = cols)


    # keep only english ingredients and ingredients with score
    eng_data = all_data[(all_data['countries'] == 'US') &
                            (all_data['ingredients_text'].notnull()) &
                            (all_data['nutrition-score-fr_100g'].notnull())].reset_index()




if __name__ == '__main__':
    main()
