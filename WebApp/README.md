oWe used shiny.io to host all files in the **WebApp Bundle** directory along with a Google OAuth file that allowed the application access to a Google account -- some sort of server solution is needed that will allow multiple people to be editing the Google Doc while retaining reasonable speed.

In order for the app to be used, the sheets need to be included as part of a Google Docs file.  

The webapp works using three Google sheets:

**settings.csv** contains the basic settings that the app uses for collecting ratings.  The Notes field in the sheet provides a description of each setting, but here is an overview of the features that one can set:

* Maximum number of ratings that a rater can do in a session.
* Number of ratings that a rater does before being asked if they want to do another round.
* Number of consecutive concepts of a type (behaviors, identities, modifiers) that a rater is given before switching to a different type.
* Whether the raters should do E, P, and A seriatim for each concept they are given (called "same phrase"), or only do a series of concepts on one dimension ("same dimension") before switching to another.
* If "same dimension": Number of consecutive ratings within a particular dimension (E,P,A) by a rater. 
* How E, P, or A should be ordered, or if their order should be random.
* Amount of money a rater will be told they are paid for a session.

Some of the settings must be evenly divisible others, as indicated in the Conditions field of the sheet:

* The maximum number of ratings must be evenly divisible by the ratings per round.
* The ratings per round must be evenly divisible by the number of ratings per block (type).
* If the "same dimension" option is specified, the number of ratings per block must be evenly divisible by the number of ratings per dimension.

While the app was developed to allow multiple rounds by a rater, with the option for them to continue, we did not field our study using this feature and so it has not been field tested for bugs.

**concepts.csv** contains the list of concepts to be fielded.  The phrase field contains the whole text of the concept, while type indicates "noun" (identity), "verb" (behavior), and "adjective" (modifier).

**results.csv** contains the results.  Each rating is on its own row, with the rating value provided, the full text of the concept, its dimension, the time from the prior rating, and the MTurk_id and IP of the rater.
