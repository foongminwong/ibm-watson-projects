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

**Exploratory Data Analysis**
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

**GroupBy in Python**
* `dataframe.groupby()`
    * can be applied on categorical variables
    * group data into categories
    * single/multiple variables
* `df_test = df[['drive-wheels','body-style','price']]`
* `df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()`
* `df_grp`
* find the average "price" of each car based on "body-style" - `df[['price','body-style']].groupby(['body-style'], as_index=False).mean()`
* `pivot()` - example: `df_pivot=df_grp.pivot(index='drive-wheels', columns='body-style')`
* pivot - one variable displayed along the columns and the other variable displayed along the rows
* heatmap - plot target variable over multiple variables
* `plt.pcolor(df_pivot, cmap='RdBu')`
* `plt.colorbar(); plt.show()`

**Correlation**
* coorelation = measure to what extent different variables are interdependent
* `sns.regplot(x="engine_size",y="price", data=df)`
* `plt.ylim(0,)`

**Correlation - Statistics**
* Pearson Correlation - measure strength of correlation between 2 features
* correlation coefficient (close to +1 - large +ve relationship,-1,0 - no relationship)
* p-value (p<0.001 - strong certainty in the result, p<0.05 - moderate, p<0.1 - weak, p > 0.1 - no certainty)
* `pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])`
* correlation heatmap

**Analysis of Variance ANOVA**
* Statistical comparison of groups
* Why: Finding correlation between different groups of a categorical variable
* What we get from ANOVA:
    * F-test score: variation between sample group means divided by variation within same group
    * p-value: confidence degree (whether the obtained result is statistically significant)
* Small F = poor correlation between variable categories and target variable
* Large F = strong correlation between variable categories and target variable (the diff between the prices of Hondas  & Jaguar is VERY significant)

## Mdoel Development

**Model Development**
* Model = a mathematical equation used to predict a value given 1/more other values (relating 1/more independent vriables to dependent variables)
    * E.g.: Input (Independent Variables/features) `highway-mpg`-> model -> Output (Dependent Variables) `predicted price` 
* The more relevant data you ahve, the more accurate your model is
* In order to getting more data you can try different types of models
    * Simple Linear Regression (SLR)
        * 1 independent var to make a prediction
        * use training points to fit or train our model & get parameters
        * store datapoints in dataframes/numpy arrays
        * use the model to predict the price of the car, BUT *model is NOT ALWAYS CORRECT* (predicted value does not match the actual value, if it's corerct, then it's due to the noise)

        * Import linear_model from scikit-learn`from sklearn.linear_model import LinearRegression`
        * Create a Linear Regression Object using the constructor `lm = LinearRegression()`
        * Predictor variable `X=df[['highway-mpg']]`
        * Target variable `Y=df['price']`
        * Fit the model `lm.fit(X,Y)`
        * Obtain prediction `Yhat=lm.predict(X)` - output as an array
        * Intercept (b0) `lm.intercept_`
        * Slope coefficient `lm.coef_`

    * Multiple Linear Regression (MLR)
        * Multiple independent vars to make a prediction (1 continuous target (Y) var & 2 or more predictor (X) vars)
        * Extarct 4 predictor vars `Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]`
        * Train model `lm.fit[Z,df['price']]`
        * Obtain prediction `Yhat=lm.predict(X)` - output = array with same # of elements as # of samples
    * Polynomial Regression and Pieplines
        * when linear model is not the ebst fit of our data
        * Polynomial Regression
            * special case of general linear regression model
            * describe curvilinear relationships - by squaring/ setting higher order terms of the predictor vars
            * Quadratic 2nd order, Cubic 3rd order, Higher order
            * Create 3rd order polynomial regression model `f=np.polyfit(x,y,3)`
            * `p=np.polyld(f)` -> `print(f)`
        * Polynomial Regression with >1 Dimension
            * `from sklearn.preprocessing import PolynomialFeatures`
            * Create a 2nd order polynomial transform object pr `pr=PolynomialFeatures(degree=2, include_bias=False)`
            * Transform the data `pr.fit_transform([1,2], include_bias=False)`
            * As dimension of data gets larger, may want to normalize multiple features in scikit-learn. We can use the preprocessing module to simplify many tasks
                * Normalize/Standardize each feature simultaneously
                * Import StandardScaler `from sklearn.preprocessing import StandardScaler`
                * Train the object `SCALE=StandardScaler()`
                * Fit the scale object `SCALE.fit(x_data[['horsepower', 'highway-mpg']])`
                * Transform the data into a new data frame n array x_scale `x_scale=SCALE.transform(x_data[[]'horsepower','highway-mpg'])`
        * Pipelines - simplify code by using pipeline library
            * Many steps to getting prediction: Normalization -> polynomial transform -> linear regression
            * Pipeline sequentially perform a series of transformations, last steo carries out prediction (LR)
            * Import all modules that we need `from sklearn.preprocessing import PolynomialFeatures` 
            * `from sklearn.lineaar_model import LinearRegression`
            * `from sklearn.preprocessing import StandardScaler`
            * Import library pipeline `from sklearn.pipeline import Pipeline`
            * Create list of tuples, the 1st element in the tuple contains the name of estimator model, 2nd element contains model constructor `Input = [('scale', StandardScaler()),('polynomial', PolynomialFeatures(degree=2)),...('mode',LinearRegression())]`
            * Input the list in the pipeline COnstructor `pipe=Pipeline(Input)` pipe is anPipeline object
            * Train pipeline by applying the train method to the pipeline obj `pipe object` `Pipe.fit(df[['horsepower', 'curb-weight','highway-mpg']], y)`
            * Produce prediction `yhat=Pipe.predict(X[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])`
            * Normalizes data -> Perform Polynomial Transform -> Output prediction

**Model Evaluation using Visualization**
* actual y, predicted yhat
* Regression Plot
* Pros: A good estimate of relaetionship between 2 vars, the strength of correlation, direction of the relationship (+ve/-ve)
* Combination of scatterplot (each point with different y) & fitted linear regression (yhat)
    * `import seaborn as sns`
    * `sns.regplot(x="highway-mpg",y="price",data=df)`
    * `plt.ylim(0,)`
* Residual Plot
    * Represents error between actual val, examine the difference between predicted val & actual val (pred - actual)
    * Expected results: Zero mean, distributed evenly around the x axis with similar variance, no curvature
    * Spread of resids: Randomly spread out around x-axis then a linear model is appropriate
    * If we see a "U" shape resid plot with not randomly spread out around the x-axis, then nonlinear model may be more appropriate
    * If we see a "<" shape resid plot (the variance of the reids increase with x), then the model is incorrect
        * `import seaborn as sns`
        * `sns.residplot(df['highway-mpg'],df['price'])`
* Distribution Plot
    * Counts the predicted val (fitted values result from model) versus actual val
        * `import seaborn as sns`
        * `ax1 = sns.distplot(df['price'], hist=False, color="r",label="Actual Value")`
        * `sns.displot(Yhat, hist=False, color="b",label="Fitted Values", ax=ax1)`
M**easures for In-Sample Evaluation**
* Determine how good the model fits on data
* 2 important emsures to determine the fit of model
    * Mean Squared Error (MSE) - (actual - predicted) ^2
        * The mean/average of all the errors by adding them together & dividing by # of samples
        * Find MSE in Python `from sklearn.metrics import mean_squared_error`
        * 2 input: actual val & predicted val of target variable `mean_squared_error(df['price'], Y_predict_simple_fit)` or `mean_squared_error(Y,Yhat)`
    * R-squared (R^2) = Coefficient of Determination
        * measure to determine how close the data is to fitted regression line
        * Percentage of variation of the target variable (Y) that is explained by the linear model
        * `R^2 = (1 - MSE of regression line/MSE of average data or ybar)`
        * `lm.score(X,Y)` if R^2 = 0.49, the approxiamtely 49.695% of the variation of price is explained by the simple linear model
        * R^2 between 0 and 1. If R^2 negative, then might be overfitting

**Prediction and Decision Making**
* Determine a Good Model Fit:
    * Make sure model results/predicted values make sense
        * Train the model `lm.fit(df['highway-mpg'], df['prices'])`
        * Predict price of car with 30 highway-mpg `lm.predict(np.array(30.0).reshape(-1,1))`
        * `lm.coef_`
        * Import numpy `import numpy as np`
        * Use numpy func arrange to generate s seq from 1 to 100 `new_input=np.arrange(1,101,1).reshape(-1,1)` - 1, 2,3...99,100 -> incremenet seq 1 step at a time
        * yhat=lm.predict(new_input)
    * Viusalization (regression Plot, resid plot, if have curvature, then suggesting nonlinear behavior, dist plot - good for multi liner reg)
        * dist plot - suggests a non-linear model may be more suitable
        * or we might need more data in this range
    * Nuemrical measures for evaluation (determine if a model is good or not)
        * MSE (As MSE increases, targets get further from the predicted points)
        * R^2
    * Comparing between different models
        * Comparing MLR and SLR
        * Is lower MSE always implying better fit?
            * Not necessarily
            * MSE for an MLR model will be smaller than the MSE for an SLR model, since the erros of the data will decrease when more variables are included in the model
            * Poly reg will also have a smaller MSE then regular regression
            * A similar inverse relationship holds for R^2

## Model Evaluation and Refinement

**Model Evaluation and Refinement**
* Tell us how model performs in the real world
* In-sample evaluation
    * how well our model will fits the data used to train it
    * Problem: Does not tell us how well the trained model can be used to predict new data 
    * Solution: 
        * Use In-sample data or training data to train the model
        * Use Out-of-sample data/test data/set, to approximate how the model performs in the real world
* Separate data into Training/Testing Sets
    * Use test data to get an idea of how our model will perform in the real world
    * Larger - training (70% of the data) , smaller - testing (30% for testing)
    * Training Set - build a model & discover predictive relationships
    * Testing set - evaluate model performance
    * After completed testing our model, should use ALL the data to train the model to get the BEST performance
    * Split datasets into random train & test subsets `from sklearn.model_selection import train_test_split`
    * `x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.3, random_state-0)`
    * x_data - independent vriables (highway-mpg,...), y_data - target (df['price']), x/y_train - parts of available data as training set, test_size - percentage of the data for testing (here 30%), random_state - # generator/random seed for random data set splitting
* Generalization Performance
    * Measure of how well our data does at predicting previously unseen data
    * Error we obtain using our testing data = approximation of this error
    * before that, we saw the distribution plot is pretty good and predicted dist fits well on actual dist, however, after seeing the dist plot using test data, the distribution becomes RELATIVELY DIFFERENT

    * Lots of Training Data
        * Accurate means of how well the model performs in the real world
        * BUT Precision performance low
    * Less Training Data, More to test model
        * accuracy of generalization performance will be less
        * BUT model has good precision
        * furtehr away from the true generalization performance (the bullspoint)
* Cross validation
    * To avoid above generalization error
    * Common out-of-sample evaluation (test) metrics
    * effective use of data (each observation is used for both training & testing)

        1. Dataset split into K equal groups (each group called as a fold, some of the folds can be used as a training set which we use to train the model & remaining parts are used as test set)
        2. Example: 3 folds for training, 1 for testing (Repeated until each partition is used for both training & testing)
        3. Use average results as the estimate of out-of-sample error
        4. Evaluation metrics depend on models such as R^2
    * Import libraries `from sklearn.model_seletion import cross_val_score`
    * `scores=cross_val_score(lr, x_data,y_data,cv=3)`
        * lr - type of model we use to do cross validation (e.g: LinearRegression model/obj)
        * x/y_data = data 
        * cv = # of partition
        * returns an array of scores, one for each partition that was chosen as the testing set
        * each calculated score is R^2
    * Average the result together to estimate out of sample R^2 `np.mean(scores)`
    * `cross_val_score` returns a score value to tell us the cross validation result. IF we want to know the actual predicted values from our model BEFORE the R^2 are calculated? 
        * Use `cross_val_predict()` - returns prediction that obtained for each element when it was in test set
            * `from sklearn.model_selection import cross_val_predict`
            * `yhat=cross_val_predict(lr2e,x_data,y_data,cv=3)`

**Overfitting, Underfitting and Model Selection**

* Model Selection
    * Underfitting - the model is too simple to fit the data
        * If we increase the order of the polynomial, the model fits better, BUT still not flexible enough and exhibits underfitting
    * Overfitting - the model is too flexible & fits the noise rather than the function, the estimated function oscillates not tracking the function
    * MSE of training & testing set of different order polynomials
        * training error ↓ with the order of polynomial
        * test error ↓ till best order of polynomial then error begins ↑
    * `Training points = y(x) + noise` noise is random, can't predict = irreducible error
    * plot of R^2 -> closer R^2 to 1, more accourate the model is
* Create an empty list to store vals `Rsqu_test=[]`
* Create list containing diff polynomial orders `order=[1,2,3,4]`
* Iterate the list using a loop

<code>
for n in order:

    # create a polynomial feature object with the order of the polynomial as a parameter
    pr=PolynomialFeatures(degree=n)

    # transform training & test data into polynomial using the fit transform method
    x_train_pr=pr.fit_transform(x_train[['horsepower']]) 

    x_test_pr=pr.fit_transform(x_test[['horsepower']]) 

    # fit regression model suing transform data
    lr.fit(x_train_pr, y_train)

    # calculate R^2 using test data and store in array
    Rsqu_test.append(lr.score(x_test_pr,y_test))

</code>

**Ridge Regression**

* Control the magnitude of those polynomial coefficients (estimated polynomial coefficients have a very LARGE magnitude) by introducing parameter alpha
* Alpha - we select b4 fitting/training a model carefully
* As alpha ↑, parameters get smaller
* If aplpha TOO LARGE, coef will appraoch 0 & underfit data, prevents overfitting, R^2 on test data ↓
* If alpha is 0, overfitting is evident
* To select alpha, we use **cross validation**
    * `from sklearn.linear_model import Ridge`
    * `RidgeModel=Ridge(alpha=0.1)`
    * `RidgeModel.fit(X,y)`
    * `Yhat=RidgeModel.predict(X)`
    * select alpha that maximizes R^2

**Grid Search**
* Scan trhough multiple free params with few lines of code/ takes the model/objs you would like to train & diff vals of the hyperparameters
* params like alpha - not part of the fitting/training process
* Hyperparameter - e.g.: alpha in Ridge regression is called hyperparameter
* **Grid Search** - scikit-learn has a means of automatically iterating over these hyperparameters using cross-validation

