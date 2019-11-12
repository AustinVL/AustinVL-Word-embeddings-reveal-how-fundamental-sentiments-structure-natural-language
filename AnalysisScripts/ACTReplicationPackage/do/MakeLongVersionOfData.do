/*
This file makes a version of the analysis dataset with one term-dimension per row,
as opposed to having one term per row and different dimensions in that row
*/

use ../derived/term_level_analysis, clear

local count = 1
foreach dim in e p a {

	foreach term in mean sd N pred {
		rename `dim'_`term' `term'`count'
	}

	foreach term in mean sd {
		rename dict_`dim'_`term' dict_`term'`count'
	}
	
	local count = `count' + 1
}

reshape long mean sd N pred dict_mean dict_sd, i(term) j(dimnum)

gen dimension = "E" if dimnum == 1
replace dimension = "P" if dimnum == 2
replace dimension = "A" if dimnum == 3
label variable dimension "E, P, or A"

label define dimnum 1 "E" 2 "P" 3 "A"
label values dimnum dimnum
label variable dimnum "E, P, or A (as number)"

saveold ../derived/term_level_analysis_long, version(12) replace
