# SportBikes-price-prediction-ML

>Name of publisher : Pranav Subhash Tikone
>Project no. : 1
>Name of project : Sport Bikes Price prediction 
>Date of Project : 12/04/2023
>Topic : Supervised Machine Learning

Description: 

This project determines the prices of the Sport Bikes based on the training data provided. 

I have used 5 features in this project for the prediction of the prices of sport bikes that are as following :

1. CC (x1)
2. Mileage (x2)
3. Max Torque (x3)
4. Max. Power (x4)
5. Fuel Tank Capcity (x5)

The program takes in input the above features of the bike and gives you the approx price of the bike which can be estimated using the provided data.
This uses MULTIPLE REGRESSION MODEL and GRADIENT DESCENT ALGORITHM, the most basic and widely used algorithm in MACHINE LEARNING for the prediction of the prices. The training set contains the input features and also the prices of those bikes. 
The algorithm runs and estimates the values of parameters W and B given an appropriate ALPHA for the GRADIENT DESCENT, then used to calculate the estimation value of the model. 

The modules used for this project are:
1. NUMPY -  to create diff arrays and work through some matrix calculations
2. MATPLOTLIB (PYPLOT) - to plot the graph between the COST vs NO. OF ITERATIONS
3. MATH - to use functions like ceil
4. COPY - to copy an entire array/matrix

I have estimated the output for 2 diff bikes, one in the training set ( to verify that the output price and the input training set price for the computed values of parameters is nearly same ) and one out of the training ( to verify the price calculated by model and the original price of that bike as mentioned on website is the same ). The results are just very accurate and the estimations made are approximately near to perfection.

Also the graph of COST vs NO. OF ITERATIONS is as expected ( Decreasing Graph as the no. of iterations increase )

However, as this contains only few features of the bikes, the prices of some bikes may vary. Thus, more the number of features, more accurately it can provide the price of the bike.
