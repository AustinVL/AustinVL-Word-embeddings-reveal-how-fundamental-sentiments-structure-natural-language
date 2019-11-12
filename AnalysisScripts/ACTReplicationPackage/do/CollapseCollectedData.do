/*
Takes cleaned ratings data and collapses to dataset in which each term is a
row with means, sd, and N for each dimension
*/

use ../derived/newratings, clear

drop if problem_rater == 1

collapse (mean) mean = rating_value (sd) sd=rating_value (count) N = rating_value, by(term kind dimension)

gen dimnum = .
replace dimnum = 1 if dimension == "evaluative"
replace dimnum = 2 if dimension == "potency"
replace dimnum = 3 if dimension == "activity"

drop dimension

reshape wide mean N sd kind, i(term) j(dimnum)

gen kind = kind1
drop kind1 kind2 kind3

foreach stat in mean sd N {
	rename `stat'1 e_`stat'
	rename `stat'2 p_`stat'
	rename `stat'3 a_`stat'
}

saveold ../derived/newratings_collapsed, version(12) replace
