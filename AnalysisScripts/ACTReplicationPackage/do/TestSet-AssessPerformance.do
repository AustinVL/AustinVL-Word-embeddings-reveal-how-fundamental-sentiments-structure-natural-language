

capture program drop GetResults
program define GetResults

frame create results str8 dim str12 group r2 mae pcen pnorm N

quietly foreach dim in e p a {

	foreach kind in overall identity behavior modifier {

		preserve
	
		if "`kind'" != "overall" {
			keep if kind == "`kind'"
		}
		
		regress `dim'_mean `dim'_pred
		local `dim'r2 = e(r2)
		local `dim'N = e(N)
	
		gen `dim'_dif = abs(`dim'_mean - `dim'_pred)
		su `dim'_dif
	
		gen `dim'_err = `dim'_mean - `dim'_pred

		local `dim'mae = r(mean)
		local `dim'N = r(N)
	
		ttest `dim'_err == 0
		local `dim'pcen = r(p)
		return list
		quietly	sktest `dim'_err
		local `dim'pnorm = r(P_chi2)
	
		noisily di "`dim' - R2  " %5.3f ``dim'r2' " (N=``dim'N')"
		noisily di "`dim' - MAE  " %5.3f ``dim'mae' " (N=``dim'N')"
		noisily di "`dim' - Center" %5.4f ``dim'pcen'
		noisily di "`dim' - Normal" %5.4f ``dim'pnorm'
	
		restore
	
		frame post results ("`dim'") ("`kind'") (``dim'r2') (``dim'mae') (``dim'pcen') (``dim'pnorm') (``dim'N')
		}

}

end


frame reset

use ../derived/training_and_new_set_predictions, clear
keep if test_set == 1

foreach dim in e p a {
	rename predicted_`dim' `dim'_pred
	rename actual_`dim' `dim'_mean
}

GetResults

frame results: list
frame results {

	capture drop *ref
	gen groupref = 1 if group == "overall"
	replace groupref = 2 if group == "identity"	
	replace groupref = 3 if group == "behavior"
	replace groupref = 4 if group == "modifier"
	
	gen dimref = 1 if dim == "e"
	replace dimref = 2 if dim == "p"	
	replace dimref = 3 if dim == "a"

	reshape wide r2 mae pcen pnorm dim group N, i(groupref) j(dimref)
	list
}


	cwf results
	
	drop in 1

	list

	format r2* mae* %9.3f
	format pcen* pnorm* %9.2f

	foreach result in r2 mae pcen pnorm {
		forvalues i = 1(1)3 {
		tostring `result'`i', gen(`result'`i'_txt) force usedisplayformat
	
		replace `result'`i'_txt = subinstr(`result'`i'_txt, "0.", "." , .)
		}
	}

	foreach result in pcen pnorm {
		forvalues i = 1(1)3 {
			replace `result'`i'_txt = "< .01" if `result'`i'< .01
			replace `result'`i'_txt = "< .001" if `result'`i' < .001
		}
	}

	sort groupref

	capture putdocx clear
	putdocx begin

	* gen dim_txt = upper(dim)
	gen group_txt = proper(group1)

	putdocx table test = data(group_txt r21_txt mae1_txt r22_txt mae2_txt r23_txt mae3_txt N1)

	putdocx table test(1, .), addrows(1, before)  valign(bottom)
	
	putdocx table test(1, 2) = ("R")
	putdocx table test(1, 2) = ("2"), append script(super)
	putdocx table test(1, 3) = ("MAE")
	putdocx table test(1, 4) = ("R")
	putdocx table test(1, 4) = ("2"), append script(super)
	putdocx table test(1, 5) = ("MAE")
	putdocx table test(1, 6) = ("R")
	putdocx table test(1, 6) = ("2"), append script(super)
	putdocx table test(1, 7) = ("MAE")
	putdocx table test(1, 8) = ("N")
	
	putdocx table test(1, .), addrows(1, before)  valign(bottom)

	putdocx table test(1, 2) = ("Evaluation"), colspan(2) 
	putdocx table test(1, 3) = ("Potency"), colspan(2)
	putdocx table test(1, 4) = ("Activity"), colspan(2)
	
	putdocx table test(., 2(1)8), halign(center)
		
putdocx save ../table/performance-testset, replace




