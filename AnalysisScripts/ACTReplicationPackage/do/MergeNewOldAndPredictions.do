

/*
Merges the new data, old words, and predictions data
*/

use ../derived/newratings_collapsed, clear

merge 1:1 term using ../derived/predictions

drop if _merge == 2
gen new_word = (_merge == 3)

drop _merge

merge m:1 term kind using ../derived/existing_ratings_tomerge.dta

gen old_word = (_merge == 3)

ta new_word old_word, missing

list if new_word == 1 & old_word == 1

** examples here are help and hunt, which are both words that had prepositions in existing dictionary
** and are therefore new words

replace old_word = 0 if new_word == 1

drop if _merge == 2

cap drop dimnum

save ../derived/term_level_analysis, replace

