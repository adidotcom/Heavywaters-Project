 # How do I test Aditya's webservice???
**Download the file titled payload.json from the Github repository**

First things first, please make sure that your machine has the httpie module installed. If it doesn't, install it using the following command on your terminal
 ```
 $ pip install httpie
 ```
 
 ### Please use the following command on your terminal to test your data and get the prediction
 ```
 $ http POST https://g1fw8iiy67.execute-api.us-east-1.amazonaws.com/dev1 < payload.json
 ```
 
