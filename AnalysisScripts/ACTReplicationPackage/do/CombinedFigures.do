
set scheme s1mono
program drop _all

program define supercombinedplot

args increment totalfreq // for error distribution histogram

local e_color = "blue"
local p_color = "cranberry"
local a_color = "green"

local e_title = "Evaluation"
local p_title = "Potency"
local a_title = "Activity"

** PREDICTED VS OBSERVED 

gen x = (uniform()*8) - 4
gen y = x
sort y
foreach dim in e p a {

	graph twoway ///
		(scatter `dim'_mean `dim'_pred, mfcolor(``dim'_color'%50) mlcolor(``dim'_color'%70) msize(small)) ///
		(line y x, lcolor(gs8) lpattern(solid) lwidth(thin)) ///
		, ///
		aspectratio(1) legend(off) ///
		xtitle("Predicted mean") ytitle("Observed mean") /// title("``dim'_title'") ///
		name(`dim'_predvsobs, replace)

} 

** ERROR DISTRIBUTION

foreach dim in e p a {
		

    gen `dim'_dif = `dim'_pred - `dim'_mean

	su `dim'_dif
	local meanerr = r(mean)

	
    twoway ///
        (histogram `dim'_dif, start(-4.55) width(.1) freq fcolor(``dim'_color'%50) lcolor(``dim'_color'%75)) ///
    , ///
    xscale(range(4.55 -4.55))xlabel(-4(2)4) xline(`meanerr', lpattern(solid) lcolor(gs12)) ///
	yscale(range(0 `totalfreq')) ylabel(0(`increment')`totalfreq') ///
	xtitle("Predicted - Observed") ///
	name(`dim'_errhist, replace) aspectratio(1)

}

** QNORM PLOT

foreach dim in e p a {
			
	qnorm `dim'_dif, mcolor(``dim'_color'%50) msize(small) ///
		name(`dim'_qnorm, replace) xscale(range(-3 3)) yscale(rang(-4.3 4.3)) ///
		ytitle("Predicted - Observed") ///
		aspectratio(1) ylabel(-4(2)4) xlabel(-3(1)3) rlopts(lcolor(gs8))

}

graph combine ///
	e_predvsobs e_errhist e_qnorm ///
	, ///
	col(1) ysize(4.5) xsize(1.5) ///
	title("     Evaluation") name(e_combined, replace) imargin(0 0 0 0)

graph combine ///
	p_predvsobs p_errhist p_qnorm ///
	, ///
	col(1) ysize(4.5) xsize(1.5) ///
	title("     Potency") name(p_combined, replace) imargin(0 0 0 0)

graph combine ///
	a_predvsobs a_errhist a_qnorm ///
	, ///
	col(1) ysize(4.5) xsize(1.5) ///
	title("     Activity") name(a_combined, replace) imargin(0 0 0 0)

graph combine ///
	e_combined p_combined a_combined, col(3) ///
	ysize(4) xsize(3.5) imargin(0 0 0 0)
	
end



use ../derived/training_and_new_set_predictions, clear
keep if test_set == 1
foreach dim in e p a {
rename actual_`dim' `dim'_mean
rename predicted_`dim' `dim'_pred
}
supercombinedplot 10 40

graph export ../fig/test_set.png, replace

use ../derived/term_level_analysis, clear
keep if new_word == 1
supercombinedplot 5 20

graph export ../fig/new_set.png, replace

exit

