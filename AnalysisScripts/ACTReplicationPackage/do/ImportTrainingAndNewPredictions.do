
import delimited ../raw/training_set_predictions.csv, clear

gen training_set = 1
label variable training_set "Part of training set?"
gen test_set = 0
label variable test_set "Part of test set?"

foreach dim in e p a {
	rename is_predicted_`dim' predicted_`dim'
}

saveold ../derived/training_set_predictions, version(12) replace

import delimited ../raw/test_set_predictions.csv, clear

gen training_set = 0
gen test_set = 1

saveold ../derived/test_set_predictions, version(12) replace

use ../derived/training_set_predictions, clear
append using ../derived/test_set_predictions

rename v1 row
label variable row "Row in original spreadsheet"

gen kind = ""
replace kind = "identity" if row >= 0 & row <= 669
replace kind = "modifier" if row >= 670 & row <= 1274
replace kind = "identity" if row == 1275 // weird case: god
replace kind = "behavior" if row >= 1276 & row < .

ta kind, missing

label data "(Predictions from training and new data)
saveold ../derived/training_and_new_set_predictions, version(12) replace

exit
