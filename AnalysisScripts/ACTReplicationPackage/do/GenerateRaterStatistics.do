/*
Generates dataset of statistics on the different raters for purpose of determining
if there are problems with ratings
*/

version 16
frames reset
program drop _all

/*
SUBPROGRAMS FOR COMPUTING INDIVIDUAL RATING STATISTICS

- assume inside a frame which only includes ratings from one rater
*/

capture program drop PositiveRatings
program define PositiveRatings, rclass

	count if rating_value > 0 & rating_value < .
	local pos = r(N)
	count if rating_value < 0 & rating_value < .
	local neg = r(N)
	local pctpos = `pos' / (`pos' + `neg')
	return scalar pctpos = `pctpos'

end	

capture program drop NearZeroRatings
program define NearZeroRatings, rclass

	syntax [, thresh(real .3)] // option to vary what counts as near zero

	count if rating_value < .
	local total = r(N)
	count if abs(rating_value) <= `thresh' 
	local nearzero = r(N)
	local pctnearzero = `nearzero' / `total'	
	return scalar pctnearzero = `pctnearzero'
	

end	

capture program drop ExtremeRatings
program define ExtremeRatings, rclass

	syntax [, thresh(real 4)] // option to vary definition of extreme 

	count if rating_value < .
	local total = r(N)
	count if abs(rating_value) >= `thresh' & rating_value < .
	local extreme = r(N)
	local pctextreme = `extreme' / `total'
	return scalar pctextreme = `pctextreme'
	
end

capture program drop VeryQuickRatings
program define VeryQuickRatings, rclass

	syntax [, thresh(real 2)] // option to vary definition of very quick 

	count if time < . & rating_value < .
	local total = r(N)
	count if time <= 2 & rating_value < .
	local quick = r(N)
	local pctquick = `quick' / `total'
	return scalar pctquick = `pctquick'
	
end

capture program drop SkippedRatings
program define SkippedRatings, rclass

	count if rating_value >= .
	local skip = r(N) 
	count if rating_value < . | rating_value >= .
	local total = r(N)
	local pctskip = `skip' / `total'
	return scalar pctskip = `pctskip'
	
end

capture program drop ModalRating
program define ModalRating, rclass

	** MODAL RATINGS
	cap drop mode
	egen mode = mode(rating_value), minmode
	count if rating_value == mode & mode < .
	local mode = r(N)
	count if rating_value < .
	local total = r(N)
	local pctmode = `mode' / `total'
	return scalar pctmode = `pctmode'
	drop mode

end

capture program drop ItemRest
program define ItemRest, rclass

	quietly {
		frlink m:1 term dimension, frame(means)
		frget mean N, from(means) 
		gen total = mean * N
		gen rest_total = total - rating_value
		gen rest_mean = rest_total / (N-1)
		cor rating_value rest_mean
	}

	return scalar itemrest = r(rho)
	
end
			
use ../derived/newratings, clear

cap drop rater_id
egen rater_id = group(mturk_id)
ta rater_id

levelsof rater_id
local rater_list "`r(levels)'"

** needed for item rest-score, but don't want to have to repeat each case in loop
frame copy default means, replace
frame means {
	collapse (mean) mean=rating_value (count) N=rating_value, by(term dimension)
	list
}

capture frame drop rater_stats
frame create rater_stats id str20(mturk_id) pctpos pctnearzero pctextreme pctquick mediantime pctskip pctmode itemrest

foreach rater in `rater_list' {	
	frame copy default onerater, replace
	frame onerater {
		quietly {

			keep if rater_id == `rater'

			** % POSITIVE RATINGS (of all non-zero ratings
			PositiveRatings
			local pctpos = r(pctpos)
			* noi di "% Positive ratings for rater `rater': " `pctpos'

			** NEAR ZERO RATINGS
			NearZeroRatings
			local pctnearzero = r(pctnearzero)
			* noi di "% ratings near zero for rater `rater': " `pctnearzero'
			
			** EXTREME RATINGS
			ExtremeRatings
			local pctextreme = r(pctextreme)
			* noi di "% extreme ratings for rater `rater': " `pctextreme'

			** VERY QUICK RATINGS
			VeryQuickRatings
			local pctquick = r(pctquick)
			* noi di "% very quick ratings for rater `rater': " `pctquick'

			** MEDIAN RATING TIME
			su time, detail
			local mediantime = r(p50)
			* noi di "median time for rater `rater': " `mediantime'
			
			** SKIPS
			SkippedRatings
			local pctskip = r(pctskip)
			* noi di "% skipped by rater `rater':" `pctskip'
			
			** MOST COMMON RATINGS
			ModalRating
			local pctmode = r(pctmode)
			* noi di "% most common rating for rater `rater': " `pctmode' 
			
			** ITEM-REST CORRELATION
			ItemRest
			local itemrest = r(itemrest)
			* noi di "item-rest correlation: " `itemrest'
			
			local mturkid = mturk_id[1]
			
			frame post rater_stats (`rater') ("`mturkid'") (`pctpos') (`pctnearzero') ///
				(`pctextreme') (`pctquick') (`mediantime') (`pctskip') ///
				(`pctmode') (`itemrest')
			
		}		
	}
}

frame rater_stats: label data "Statistics about performance of each rater"
frame rater_stats: saveold ../derived/individual_rater_stats.dta, replace version(11)



