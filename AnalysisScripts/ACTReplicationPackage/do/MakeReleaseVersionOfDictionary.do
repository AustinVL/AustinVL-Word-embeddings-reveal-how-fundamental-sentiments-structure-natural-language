use ../derived/term_level_analysis, clear

rename term concept

label variable new_word "New concept or one repeated from US-Online Dictionary?"

foreach dim in e p a {
	if "`dim'" == "e" {
		local text "Evaluation" 
	}
	if "`dim'" == "p" {
		local text "Potency" 
	}
	if "`dim'" == "a" {
		local text "Activity" 
	}
	label var `dim'_mean "`text' mean rating"
	label var `dim'_sd "`text' rating sd"
	label var `dim'_N "`text' number of ratings"
}

drop _merge
drop old_*
drop *_pred
drop dict*

order concept kind new_word

saveold ../derived/EPA_newdata_dictionary, version(12) replace

export delimited using ../derived/EPA_newdata_dictionary, replace
