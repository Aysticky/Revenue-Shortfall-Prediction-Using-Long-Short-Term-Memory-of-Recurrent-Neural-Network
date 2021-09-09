%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project carried out under machine learning COEN 807%
% Olusesi Ayobami Meadows                            %
% meadowsolusesi@gmail.com; oameadows@abu.edu.ng     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% saved data in variable1
data = VarName1;

figure
plot(data)
xlabel('Hour')
ylabel('Shortfall')
title('Hourly shortfall of Electricity Revenue')


numTimeStepsTrain = floor(0.9*numel(data));

% split into train (first 90%) and test (remaining 10%)

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

% standardize the data for zero mean and unit variance

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

% to understand how to get x and y from dataTrainStandardized
% consider [1,2,3,4,5] we can split it in number of ways to 
% predict next integer which is 6
% first: x  y
%      [1,2] 3
%      [2,3] 4
%      [3,4] 5
%      [4,5] 6
%      
% second: x      y
%      [1,2,3]   4
%      [2,3,4]   5
%      [3,4,5]   6

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% Xtrain (x)    Ytrain (y)
% [1,2,3,4]     [2,3,4,5]
% [2,3,4,5]     [3,4,5,6]
% [3,4,5,6]     [4,5,6,7]

% after creating Xtrain and Ytrain, we create four layers
% and set certain parameters, then train the network

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [sequenceInputLayer(numFeatures),lstmLayer(numHiddenUnits),fullyConnectedLayer(numResponses),regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');


net = trainNetwork(XTrain,YTrain,layers,options);

% standardize the test data using same mean and 
% standard deviation

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);

% make the first prediction using the last time step
% of training response: YTrain (end)

[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);

% use previous prediction as an input in for loop

for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end


YPred = sig*YPred + mu;


YTest = dataTest(2:end);
% calculate root mean square error
rmse = sqrt(mean((YPred-YTest).^2))

% plot training data with the forecasted values
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel('Hour')
ylabel('Shortfall')
title('Forecast')
legend(['Observed' 'Forecast'])

% compare the forecasted values with the test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(['Observed' 'Forecast'])
ylabel('Shortfall')
title('Forecast')

% subplot(2,1,2)
% stem(YPred - YTest)
% xlabel('Hour')
% ylabel('Error')
% title('RMSE = ' + rmse)

% Reset the network and initialize it by predicting on XTrain
net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];

% Till this we have not use XTest, now we will use XTest instead
% of YPred in the same for-loop as previous one

numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end


YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2))

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(['Observed' 'Predicted'])
ylabel('Shortfall')
title('Forecast with Updates')

% subplot(2,1,2)
% stem(YPred - YTest)
% xlabel('Hour')
% ylabel('Error')
% title('RMSE = ' + rmse)
