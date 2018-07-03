# [no-code-ml.com](http://no-code-ml.com)
### Project Introduction
This is a personal project to create a web app that allows someone to upload different csv files with registration details (eg name, contact details etc) and import into a fixed format database.  This could be useful for a company that receives lists of customers or members in different csv structures and needs to enter this data into their own database.

Note that the web app is intended as a conceptual project rather than a commercial product. It aims to showcase the programmer's skillset of data pipeline management and web dev.

### Functionality  
The web app offers the following functionality:
* upload csv files
* view initial rows (equivalent to df.head() in pandas)
* change column names
* select columns for regression analysis
* select columns using jquery (without page refresh)
* compare regressions with different tools (linear, random forest, etc)
* demo mode

### Future Functionality
* save resulting datasets and regression analyses
* manually delete all saved files at end of session
* algorithm tuning: hyperperameters adjusted (via dropdowns/checkboxes) to optimzie models
* predictive analysis: upload unlabeled data and predict dependent variable
* missing data (NaN) handling
* multiple header handling

### Structure
* [app.py](https://github.com/howardvickers/no-code-ml/blob/master/src/app.py) is the server.  
* [index.html](https://github.com/howardvickers/no-code-ml/blob/master/src/templates/index.html) is the initial html page (includes upload function).  
* [ml.html](https://github.com/howardvickers/no-code-ml/blob/master/src/templates/ml.html) is the machine learning html page (main interface).  
* [regressions.py](https://github.com/howardvickers/no-code-ml/blob/master/src/regressions.py) runs regressions according to selected models and returns RMSE and R-Squared stats.   
* [ols_summary.py](https://github.com/howardvickers/no-code-ml/blob/master/src/ols_summary.py) runs initial linear regression and initial random forest to generate coefficients, p-values and feature importances (to assist with feature/variable selection).   
* [global_y.py](https://github.com/howardvickers/no-code-ml/blob/master/src/global_y.py) holds the y variable (for app.py).   

### Technologies Employed
* python
* numpy
* pandas
* flask
* jinja
* javascript
* jquery
* ajax
* bootstrap
* html
* css
# uploadcsv
