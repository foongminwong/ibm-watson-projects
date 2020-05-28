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
    * `df["city-mpg"] = 235/df["city-mpg"]`
    * `dr.rename(columns={"city-mpg":"city-L/100km"}, inplace=True)`
    * Incorrect data types
    * Identify data types = `dataframe.dtype()`
    * Convert data types = `dataframe.astype()` (e.g.: `df["price"] = df["price"].astype("float")`) - cast the column price to *float*

3. Data Normalization (centering/scaling) - uniform the features value with different range
    * Not normalized -> normalized (similar value range & similar intrinsic influence on analytical model)
    * Methods of normalizing data:
        * Simple Feature Scaling: Divides each val by max val, new wal range between 0 and 1
            * xnew = xold/xmax
            * `df["length"] = df["length"]/df["length"].max()`
        *  Min-max: new val ranges between 0 and 1 
            * xnew = (xold-xmin)/(xmax - xmin)
            * `df["length"] = (df["length"]-df["length"].min())/(df["length"].max()-df["length"].min())`
        * z-score/standard score: new val range from -3 to +3
            * xnew = (xold - mu)/sigma
            * `df["length"] = (df["length"]-df["length"].mean())/df["length"].std()` price -> low price, medium price, high price instead of 5000, 10000,12000, 30000...
            * `bins = np.linspace(min(df["price"]), max(df["price"]), 4)` - return arrays of bins with 4 equally spaced numbers over the specified interval of the price
            * `group_names=["Low","Medium","High"]`
            * `df["price=binned"] = pd.cut(df["price"],bins, labels=group_name,include_lowest=True)` - use `pandas` function `cut` to segment & sort the data values into bins
            * then use histogram to visualize data dist after binning
4. Data Binning
    * group values into `bins` - group age into [0 to 5], [6 to 10], [11 to 15]
    * binning can sometimes improve the accuracy of predictive models
    * group a set of numerical values into a set of `bins` to ahve better understanding of the data distribution - converts numeric to categorical variables
    * 

5. Turning Categorical values to numeric values - make statistical modeling easier
    * Problem: Most statistical models cannot take in objects/strings as input, for model training only take numbers as inputs
    * **Solution: Add dummy variables for each unique category & Assign 0 or 1 in each category** - one-hot encoding
    * e.g. fuel: gas/diesel
    * `pandas.get_dummies()` - convert categorical variables to dummy variables ( 0 or 1)
    * `pd.get_dummies(df['fuel'])`

## Exploratory Data Analysis (EDA)

# Exploratory Data Analysis
* Summarize main characteristics of data
* Gain better understanding of the data set
* Uncover relationships between variables
* Extract important variables
* Going to learn:
    * Descriptive statistics = short summaries about the sample & measures of the data
        * `df.describe()`
        * `drive_wheels_counts = df["drive-wheels"].value_counts()` - summarize categorical data using `value_counts()` method
        * `drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)`
        * `drive_wheels_counts.index.name='drive-wheels'`
        * Boxplots - easily sport outliers, see distribution & skewness of the data
        * Scatterplot
            * each obs represented as point
            * show relationship between 2 variables
    * GroupBy
    * ANOVA
    * Correlation
    * Correlation - Statistics, Pearosn correlaation, correlation heatmaps

