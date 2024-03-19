import pandas as pd

import seaborn as sns

df = pd.read_excel(r'C:\Users\gokul\Desktop\Skill\PYTHON\OnlineRetail (1).xlsx')



# data description

df = pd.DataFrame(df)
print(df.describe())


# pivot table
df_items = df_purchase.pivot_table(index='InvoiceNo', columns=['Description'], values='Quantity').fillna(0)
df_items.head(3)


# finding the most popular products by global wise

product_counts = df.groupby('Description')['Description'].count()
most_popular_products = product_counts.idxmax()
print("The most popular products globally are:", most_popular_products)


# finding the most popular product by region or country wise

region_product_counts = df.groupby(['Country', 'Description'])['Description'].count()
most_popular_products = region_product_counts.groupby('Country').idxmax()
print("The most popular products by Country are:")
for Country, product in most_popular_products.items():
    print(Country, ":", product[1])


# finding the most popular product by month wise

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceDate'] = df['InvoiceDate'].dt.month
month_product_counts = df.groupby(['InvoiceDate', 'Description'])['Description'].count()
most_popular_products = month_product_counts.groupby('InvoiceDate').idxmax()
print("The most popular products by month are:")
for InvoiceDate, product in most_popular_products.items():
    print("InvoiceDate", InvoiceDate, ":", product[1])


df.info()
df_purchase= df[['InvoiceNo', 'StockCode', 'Description', 'Quantity']]
df_purchase.head()



df.groupby('Description').agg(
    orders=('InvoiceNo', 'nunique'),
    quantity=('Quantity', 'sum')
).sort_values(by='orders', ascending=False).head(10)


# using recommendation function(correlation)
def get_recommendations(df, item):
    recommendations = df.corrwith(df[item])
    recommendations.dropna(inplace=True)
    recommendations = pd.DataFrame(recommendations, columns=['correlation']).reset_index()
    recommendations = recommendations.sort_values(by='correlation', ascending=False)
    return recommendations


recommendations = get_recommendations(df_items, 'WHITE HANGING HEART T-LIGHT HOLDER')
recommendations.head()

recommendations = get_recommendations(df_items, 'REGENCY CAKESTAND 3 TIER')
recommendations.head()

recommendations = get_recommendations(df_items, 'JUMBO BAG RED RETROSPOT')
recommendations.head()

recommendations = get_recommendations(df_items, 'PARTY BUNTING')
recommendations.head()

recommendations = get_recommendations(df_items, 'LUNCH BAG RED RETROSPOT')
recommendations.head()

recommendations = get_recommendations(df_items, 'ASSORTED COLOUR BIRD ORNAMENT')
recommendations.head()

recommendations = get_recommendations(df_items, 'SET OF 3 CAKE TINS PANTRY DESIGN')
recommendations.head()

recommendations = get_recommendations(df_items, 'PACK OF 72 RETROSPOT CAKE CASES')
recommendations.head()

recommendations = get_recommendations(df_items, 'NATURAL SLATE HEART CHALKBOARD')
recommendations.head()

recommendations = get_recommendations(df_items, 'LUNCH BAG  BLACK SKULL.')
recommendations.head()


