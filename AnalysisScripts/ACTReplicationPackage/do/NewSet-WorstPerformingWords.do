frame reset

use ../derived/term_level_analysis, clear
keep if new_word == 1

foreach dim in e p a {
rename `dim'_mean actual_`dim'
rename `dim'_pred predicted_`dim'
}

quietly foreach dim in e p a {
    regress actual_`dim' predicted_`dim'
    local `dim'R2 = e(r2)
    local `dim'N = e(N)
    cap drop `dim'dif
    gen `dim'_dif = abs(actual_`dim' - predicted_`dim')
    su `dim'_dif
    local `dim'mae = r(mean)
    noisily di "`dim' -- R2 " %6.3f ``dim'R2' " MAE " %6.3f ``dim'mae' " (N=``dim'N')"
}

foreach dim in e p a {
    gsort -`dim'_dif
    di upper("`dim'")
    list term kind predicted_`dim' actual_`dim' `dim'_dif in 1/10
	frame copy default `dim'_result, replace
}

foreach dim in e p a {
	frame `dim'_result {
		keep in 1/10
		list term kind predicted_`dim' actual_`dim' `dim'_dif
		gen dim = "`dim'"
	}
	
}

frame e_result {

	cap drop _merge
	frame p_result: cap drop _merge
	frame a_result: cap drop _merge
	
	frameappend p_result
	frameappend a_result

	replace term = proper(term)
	replace kind = proper(kind)
	
	foreach word in predicted actual {
		gen `word' = .
		replace `word' = `word'_e in 1/10
		replace `word' = `word'_p in 11/20
		replace `word' = `word'_a in 21/30
	}

	gen difference = predicted - actual
	
	format predicted actual difference %7.2f
	
	list term kind predicted actual difference

	capture putdocx clear
	putdocx begin
	putdocx table test = data(term kind predicted actual difference)
	
	putdocx table test(21, .), addrows(1, before)  valign(bottom)	
	putdocx table test(21, 1) = ("Activity")
	putdocx table test(11, .), addrows(1, before)  valign(bottom)	
	putdocx table test(11, 1) = ("Potency")
	putdocx table test(1, .), addrows(1, before)  valign(bottom)	
	putdocx table test(1, 1) = ("Evaluation")

	putdocx table test(1, .), addrows(1, before)  valign(bottom)	
	putdocx table test(1, 1) = ("Concept")
	putdocx table test(1, 2) = ("Type")
	putdocx table test(1, 3) = ("Predicted")
	putdocx table test(1, 4) = ("Actual")
	putdocx table test(1, 5) = ("Difference")
	
	putdocx save ../table/NewSet-WorstWords, replace

}

frame reset
