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

**Getting Started Analyzing Data in Python**
* why check data types using `df.dtypes`?
    * potential info & type mismatch
    * compatibility with python medthods
* `df.describe()` = returns a statistical summary (count, mean, std, min, max, quartiles), `df.describe(include="all")` = full summary statistics (+ unqiue, top, freq)
* `NaN` - not a number
* `df.info()`

**Accessing Databases with Python**
* DB-API = standard API for accessing relational databases


## Data Wrangling

**Pre-processing Data in Python**
* Data pre-processing/ data cleaning/data wrangling = process of converting/mapping data from the 'raw' form into another fofrmat in order to prepare the data for further analysis 
1. Identify and handle missing values
    * Check with the data collection source
    * Drop the missing values
        * drop the variable
        * drop the single entry of the missing value = don't have a lot of observations with missing data
        * `dataframes.dropna(subset=['column_name'],axis=0,inplace=True)` -`axis=0` that drops the entire row or `axis=1` that drops the entire column
        * checkout http://pandas.pydata.org/
    * Replace missing values - (better) no data is wasted, but less accurate since we replace the missing data with a gues of what the data should be 
        * `dataframes.replace(missing_val,new_val)`
        * Numeric variable - replace it with an average value of entire variable = 
            * `mean = df['normalize-losses'.mean`
            * `df['normalized-losses'].replace(np.nan, mean)` 
        * Categorical variable (e.g. fuel type) - replace by frequency with the mode of thte particular column/ appears msot often
        * replace it based on other functions
        * leave it as missing data 
2. Data Formatting
3. Data Normalization (centering/scaling)
4. Data Binning
5. Turning Categorical values to numeric values - make statistical modeling easier