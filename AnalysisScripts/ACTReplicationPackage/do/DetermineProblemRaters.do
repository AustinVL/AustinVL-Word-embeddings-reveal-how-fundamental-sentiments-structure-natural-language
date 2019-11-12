



/*
We follow criteria of calculating a "trouble score" for each rater based on
different dimensions of concern about their ratings.  Raters were dropped
with trouble scores of 2 or more (13 of 188).  Our criteria mostly followed
that of the data collection generating our training data, with two exceptions.
First, the data collection for the training set asked about time spent living 
in the United States and used that as one criterion.  Although we did use US IP-location
as a restriction, we did not ask raters for any personal information, including '
nativity. In computing the trouble score, we instead substituted low item-restscore correlations as an indicator of
respondents who gave unusual responses relative to others.  Second, as we were
collecting fewer ratings per respondent, we were more concerned about the
potential effects of a few individuals who gave many extreme responses (compared 
to the median 5% extreme responses among raters), and so we included this as a 
trouble criterion.  These decisions about exclusions were made prior to looking
at whether or not they improved the fit of our data either to the ratings of existing words
or our predictions about new words.


+1 if > than 85% in a given direction
+1 if > than 40% extreme ratings (+2 if > 75%)
+1 if > than 85% of ratings within .3 of zero
+1 if > than 35% of ratings skipped (+2 if >70%)
+1 if median time is < 2.25 secs (+2 if < 2 secs)
+1 if item-rest correlation is < .4 (+2 if < .1)
*/

use  ../derived/individual_rater_stats.dta, clear

	gen trouble_start = 0
	gen trouble_dir = (pctpos >= .85 | pctpos <= .15)
	gen trouble_ext = (pctextreme > .4) + (pctextreme > .75)
	gen trouble_nearzero = (pctnearzero > .85)
	gen trouble_quick = (mediantime < 2) + (mediantime < 2.25)
	gen trouble_skip = (pctskip > .35) + (pctskip > .70)
	* gen trouble_mode = (pctmode > .25)
	gen trouble_itemrest = (itemrest < .1) + (itemrest < .5)
	gen trouble_finish = 0
	egen trouble_total = rsum(trouble_start-trouble_finish)
	ta trouble_total
	table trouble_total, c(mean itemrest)
	
drop if trouble_total < 2
levelsof id
local badraters = "`r(levels)'"
di "bad raters: `badraters'"

keep id mturk_id trouble_total

saveold ../derived/problem_rater_ids, version(11) replace

exit
