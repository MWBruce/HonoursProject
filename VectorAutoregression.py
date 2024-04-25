from statsmodels.tsa.vector_ar.var_model import VAR

data = pd.read_csv('data.csv')

model = VAR(data[['price', 'book value', 'revenue per share']])
results = model.fit(maxlags=15, ic='aic')
print(results.summary())
