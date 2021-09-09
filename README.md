# Revenue-Shortfall-Prediction-Using-Long-Short-Term-Memory-of-Recurrent-Neural-Network
This project was carried out under machine learning course (COEN807). A Long-Short term memory of recurrent neural network was developed to forecast electricity income shortfalls of Canteen Feeder, Kaduna electricity company Nigeria. 

Revenue forecasting is a technique that can used by electric utilities or by energy management system (grid operator) to predict the electricity revenue shortfall needed to balance the generation and load demand.
The shortfall dataset used for this project was obtained from Kaduna Electricity Company.

Methodology

Step 1: Partition the training and test data
The dataset consists of 744 samples. data was split into train (first 90%) and test (remaining 10%). 670 instances were first trained on the sequence and next 74 instances for testing.

Step 2: Standardize the data for zero mean and unit variance.
For a better fit and to prevent the training from diverging, the training data was standardized to have zero mean and unit variance.

Step 3: Create Xtrain and Ytrain
To forecast the values of future time steps of a sequence, the responses to be the training sequences was specified with values shifted by one-time step. That is, at each time step of the input sequence, the LSTM network learned to predict the value of the next time step.

Step 4: Set parameters to train the network
The LSTM layer was specified to have 200 hidden units, 1 sequence input Layer, and 1 fully connected Layer. The network was trained for 300 epochs with adam solver. Initial learn rate of 0.005 was specified and learn rate were dropped with a multiplying factor of 0.2 after 150 epochs.

Step 5: Make the first prediction and update network with the actual values.
First, the network state was initialized by predicting on XTrain. To make predictions on a new sequence, the network state was reset using resetState. Resetting the network state prevents previous predictions from affecting the predictions on the new data. Then XTest was used instead of YPred in the same for-loop as previous one. All trained data are stored in 'variable1'.

RESULTS

![image](https://user-images.githubusercontent.com/68459726/132748066-030aac9b-67b7-45b7-b0c3-5ba5f9eb27ab.png)

Figure 1: Training data with the forecasted values
Figure 1 illustrates the fitness of the forecasted output with the trained data. 670 samples were trained and remaining 74 samples were forecasted. It is observed that the forecasted values learn quite well from the trained dataset.

![image](https://user-images.githubusercontent.com/68459726/132748526-d2eb7464-cb00-4849-9e38-a824a045fcd7.png)

Figure 2: Comparison of forecasted values with the test data 
Figure 2 shows the combination of the actual values (observed) and the forecasted values. To obtain more accurate value, the output was updated with Xtest.

![image](https://user-images.githubusercontent.com/68459726/132748294-01d5421b-8719-49f8-bf83-c1adf515c4cf.png)

Figure 3: Updated Forecasted values
In Figure 3, predictions appeared to be more accurate after updating the network state with the observed values (Xtest) instead of the predicted values; leading to a better fit than that of Figure 2.
