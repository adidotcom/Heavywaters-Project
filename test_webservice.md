 # How do I test Aditya's webservice???

```
curl -X POST  -H "Content-Type: application/json; charset=utf-8" https://g1fw8iiy67.execute-api.us-east-1.amazonaws.com/dev1 -d "{\"data\":\"enter your your test case here\"}"
```
#                                                            OR


**Download the file titled payload.json from the Github repository**

First things first, please make sure that your machine has the httpie module installed. If it doesn't, install it using the following command on your terminal
 ```
 $ pip install httpie
 ```
 
 ### Please use the following command on your terminal to test the sample document
 ```
 $ http POST https://g1fw8iiy67.execute-api.us-east-1.amazonaws.com/dev1 < payload.json
 ```
 
