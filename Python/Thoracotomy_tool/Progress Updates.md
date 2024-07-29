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