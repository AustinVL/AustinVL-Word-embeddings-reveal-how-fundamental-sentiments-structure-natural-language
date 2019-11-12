

/*
Takes the .csv of the UGA dictionary delivered to us and reduces to the words
that will allow it to merge with the collected data and predictions
*/

import delimited ../raw/UGA_EPA_dict.csv, clear
describe

replace concept = "a murderer is" if concept == "a muderer is" // error in one case

gen word_count = wordcount(concept)
ta word_count

gen word1 = word(concept, 1)
drop if word1 == "God" // weird case that is not relevant for us
ta word1

cap drop kind 
gen kind = "identity" if word1 == "a" | word1 == "an"
replace kind = "behavior" if word1 == "to"
replace kind = "modifier" if word1 == "being" | word1 == "feeling"
ta word1 kind, missing

cap drop lastword
gen lastword = word(concept, -1)
ta lastword
list concept if lastword != "is"

cap drop wordcount
gen wordcount = wordcount(concept)
ta wordcount
drop if wordcount > 5

** drop multi-word existing words
drop if wordcount > 3 & (word1 == "a" | word1 == "an")
drop if wordcount > 3 & word1 == "being"
drop if wordcount > 3 & word1 == "feeling"

cap drop term
gen term = word(concept, 2) // only ones that need to merge?
duplicates list term kind
duplicates tag term kind, generate(dupflag)
list concept term kind if dupflag != 0
duplicates drop term kind, force

rename concept_frequency dict_N
label variable dict_N "Number of ratings"

keep e_* p_* a_* dict_N term kind

foreach var of varlist e_* p_* a_* {
	rename `var' dict_`var'
}

order term kind

** TO DO: make sure dupflag == 1 does not match anything in data

label data "DO NOT USE EXCEPT TO MERGE (only includes second words of prompt)"
saveold ../derived/existing_ratings_tomerge, replace version(12)

