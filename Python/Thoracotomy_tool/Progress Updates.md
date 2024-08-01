# RTcotomy ML Application Progress Updates

##07/18/2024
- Initialized the machine learning code and made initial models with the data
- Generated HTML code to make user interface of the website
- Created application itself with python
- Use ngrok to host the website application, but have not figured out how to host a live server for the website in order

## 07/20/2024
- Started adding multiple website pages/ tabs (HTML)
- Created AWS account, app and environment
	- This seems to have not been successful, unable to get it up and running due to unknown errors
	- This may need to be done in order to get a website that runs without my compputer being the host
		- Unless there is a way to get ngrok to be the host
- A major goal needs to be that the number of samples being fed into the algorithm are quite small (~2k), so we need to add the `old data` because it would be reasonably simple to do so given the few number of data fields that we would need to collect in order to feed the algorithm.
- Another analysis that would be cool prior to the meeting on the 31st would be to show

## 07/26/2024
- Can just use predict_proba value x 100 to get a probaility associated with prediction. Should be easy to implement, will try it out.
- Successfully implemented a confidence percentage.
- Created data for algorithm using data from older stuff but does not include prehospital cardiac arrest, so will have to train that model separately and see how it does separately.

## 08/01/2024
- Tried to make it so that the app would auto update as users input data and instantly get an output. In order to maximize the number of cases for each model and not have to use imputation, I tried to train models (hyperparameters and everything) for every combination of features using EMS, ED or EMS_ED set of features. after around 66 hours of run time on my mac I killed this process as it was obviously taking too long and not a productive use of time, although it did generate the appropriate outputs for what it did finish creating. The main issue is that after this I calculated the number of combinations of features, feature sets, hyperaparameters and other variables to find that based on the current run time and progress of the script that there was a total estimated combination list of 275,850 combinations, of which in this time only around 40-50k had been completed and the projected runtime would be a total of 9.7 days...
- Needless to say, it sucks not having the MSU supercomputer anymore and am stuck with my 4 core MacbookPro
- So I undertook the process of copying over all of this stuff to my PC Desktop (will have to keep everything talking nicely between Mac and PC using GitHub). Once I have this setup and running at full capacity on my PC, with the hardware that it has, I should be able to get the run time down to approximately 2 days (still freaking crazy but the number of models being made and cross validated etc is also enormous). This should make things pretty intuitive on the app, however, because once you put in 2 pieces of information you can start to get output predictions and probabilities. 
- For now until my PC finishes running all of these remaining models, I will be able to simply use EMS or ED as separate methods of predictions, the combined will come later. It will be interesting to see if adding what is essentially time series data will influence the performance to get into the >0.9 AUROC range or if we will be stuck below 0.9. On some of my earlier attempts I was able to get over 0.9 using >100 features but they would not have been practical in any manner, so as a result, keeping the feature set smaller is a sacrifice being made for practical use
- Also, this data was trained using vitals and such from 2007 - 2020 (for some absolutely crazy reason starting in 2021 the vitals are no longer tracked from EMS in the NTDB and you have to use a NEMESIS identifier to go track all of that down, no idea how that works, could try and figure it out but it would only be for two years of data and I imagine it is an absolute royal pain in the ass) --> better to use my time for my actual project since this is really just a fun side project.
- Could get the EMS_ED part working fine if I just choose only logistic regression and speed run those models for now, xgb is better but takes wayyyy too long and too much space. Happy solution I think for time being and then can go full xgb after the meeting when I can devote more time to getting it running round the clock on PC
- Get back to working other project:
	- Need to make plot of various numerical outcomes (other than mortality) vs age groups, mortality vs other numerical outcomes, try to get old data to work for mortality stuff (got it to work with vitals, now just need the rest if possible, use the list of desired datafields), make a graph with years on the x axis and y axis is all of the different datafields to show when each come and go in the records (translate datafields that have the same data but make name changes over the years to the same thing, we want to know what info we gain/lose over time, not how we named it (could have that as a graph too just to show complicated it is though)). Can subset by procedures, hospital events etc to find out if the mortality of those groups that (received intracardiac epi, cross clamp etc) had better or worse outcomes. Add line of overall, blunt and penetrating mortality to the graphs with mortality vs age/ year/ other x axis variable to demarcate where more substantial deviations are occurring. 
	- Will need to figure out all of the appropriate stats testing, don't do anything yet because why waste time on doing all of this until we know what we  want to include in the paper with certainty and also can get input at meeting on 8/6/2024 (maybe attendings have ideas about what stats tests to use, probably present a few and ask for input).
	- Refer back to the Loyola paper to get ideas for other figures/ tables to generate. Also try to replicate their EXACT data and figures/ tables, not just using the way I do it, want to see if my method is substantially different or not, from what I can tell so far that is not the case but the earlier data had some wonky stuff going  on.
	- Need to work on incoporating ISS and AIS code stuff to see how much more detailed I can get it to be, able to make heatmap on human figure of where most lethal??? (very cool if possible).
