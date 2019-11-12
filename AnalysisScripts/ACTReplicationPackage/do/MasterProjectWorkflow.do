

version 16
clear all
frames reset

** MAKE VERSION OF DATA WITHOUT REAL MTURK IDS FOR REPLICATION PACKAGE

*	do MakeDataWithoutMTurkIDs // exclude this file from replication package

** IMPORT AND SAVE NEW RATINGS DATA
		
	import delimited ../raw/collected_epa_ratings_noid.csv, clear 
	
	gen term = word(phrase, 2)

	cap drop kind 
	gen word1 = word(phrase, 1)
	gen kind = "identity" if word1 == "a" | word1 == "an"
	replace kind = "behavior" if word1 == "to"
	replace kind = "modifier" if word1 == "being" | word1 == "feeling"
	drop word1
	
	destring rating_value, replace force
	
	replace mturk_id = upper(mturk_id)
	
	saveold ../derived/newratings.dta, replace version(12)

** IMPORT AND SAVE PREDICTION DATA

	import delimited ../raw/predictions.csv, encoding(ISO-8859-2) clear 
	
	* "oversee" and "salute" repeated in .csv for whatever reason -- delete one
	duplicates report term
	duplicates drop term, force 

	rename e e_pred
	rename p p_pred
	rename a a_pred
	
	saveold ../derived/predictions, replace version(12)

** CREATE SUBSET OF TRAINING DICTIONARY THAT WILL MERGE WITH NEW DATA

	do ProcessExistingRatings.do

** DETERMINE AND DROP PROBLEM RATERS FROM NEW DATA	
	
	do GenerateRaterStatistics.do

	do DetermineProblemRaters.do

	use ../derived/newratings.dta, clear
	merge m:1 mturk_id using ../derived/problem_rater_ids	
	assert _merge == 1 | _merge == 3 // if _merge == 2 then an mturk_id mismatch
	gen problem_rater = (_merge == 3)
	replace problem_rater = 1 if ///
		mturk_id == "A2T1K94BLNDBAH" | ///
		mturk_id == "ARJ6SPCZJVARE" | ///
		mturk_id == "AV5DCU1VH5S34"
	ta problem_rater
	drop _merge
	save ../derived/newratings.dta, replace
				
** CREATE DATASET THAT COLLAPSES NEW DATA INTO SUMMARY STATISTICS

	do CollapseCollectedData
	
** COMBINE NEW DATA WITH PREDICTIONS AND EXISTING RATINGS

	* wide version of dataset -- one row per term
	do MergeNewOldAndPredictions

	* long version of dataset -- one row per term-dimension (ie 3 rows per term)
	do MakeLongVersionOfData
	
** MAKE DATASET OF TRAINING AND TEST SET PREDICTIONS 

	do ImportTrainingAndNewPredictions.do
	
** TABLES ASSESSING PERFORMANCE

	do TestSet-AssessPerformance.do
	
	do NewSet-AssessPerformance.do

** TABLES OF WORST PERFORMING WORDS

	do TestSet-WorstPerformingWords.do

	do NewSet-WorstPerformingWords.do
	
** FIGURES WITH TEST SET AND NEW SET

	do CombinedFigures.do	
	
** MAKE RELEASE VERSION OF THE NEW DATASET

	do MakeReleaseVersionOfDictionary.do
	

	
