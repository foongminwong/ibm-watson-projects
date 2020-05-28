# Data Analysis with Python

## Importing Datasets

**Python Packages for Data Science**
* Python Scientific Computing Libraries:
    * Panadas: tools for effective data manipulation & analysis, offers data structure
        * Dataframe = 2 dimensional table consisting of row and column (easy indexing functionality)
    * NumPy: use arrays for inputs & outputs, can be extended to objects for matrices, perform fast array processing
    * SciPy: data visualization
* Python Visualization Librabries:
    * Matplotlib - highly customizable graphs/plots
    * Seaborn: based on Matplotlin, easy to generate heat maps, time series, violin plots
* Python Algorithmic Libraries in Python
    * Use machine learning algorithms able to develop mdoel using our data set & obtain predictions
        * Scikit-learn: tools for statistical modeling - regression, classification, clustering (built on NumPy, SciPy, Matplotlib)
        * Statsmodels: explore data, estimate statistical models, perform statistical tests

**Importing and Exporting Data in Python**
* Import data
    * Format: .csv (might have header in the first row, if not, each row is 1 data point), .json, .xlsx, .hdf
    * Filepath
* Import csv in Python
    * use Pandas `read_csv(?,header=None)` into panda dataframe
* Print dataframe
    * `df` = entire dataframe (can be large datasets!)
    * `df.head(n)` = show the 1st n rows of df
    * `df.tail(n)` = show bottom n rows of df
* Add headers
    * `headers = ["name", "date", ....]`
    * `df.colums = headers`
* Export Pandas df to csv
    * `df.to_csv(path)`

