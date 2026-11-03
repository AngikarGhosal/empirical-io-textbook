# GMY_Replication/Data

This file contains the raw data needed to run the estimation. The files are: 


## Primary Data Files
* `data_blp.csv` - Cleaned data on model sales, prices and characteristics for all years from Wards and supplemented from other sources described in the paper. 
* `microMomentsCEX.mat` - Collection of computed micromoments from the Consumer Expenditure Survey. 
* `microMomentsMRI.mat` - Collection of computed demographic moments from MRI Survey (underlying data proprietary).
* `secondChoiceOut.mat` - Collection of second choice correlations computed from Maritz Survey data for selected years (underlying data proprietary)
* `full_bootstrap_results.mat` - Collection of demographic moments from 500 bootstrap samples of the CEX and MRI microdata (underlying data proprietary). 
* `secondChoice_bootstrap.mat` - Collection of second choice moments from 500 bootstrap samples of Martiz Survey microdata (underlying data proprietary).
* `CPS_households.csv` - Current Population Survey data on national demographics used to simulate distribution of consumer characteristics. 


## Miscellaneous Data 
* `make_group_ids.csv` - Assigns makes to ownership groups. 
* `CPIU_1967base` - US CPI-U Index.
*  `income_quintile_cutoffs_by_year.csv` - Annual household income quintile cutoffs for each year in sample. 
