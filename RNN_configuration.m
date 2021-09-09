%%configure RNN

%1 hidden layer of 10 units,2 time step
net = recurrent(1:n,3:5);
% prepare input to get time series for training
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
%train the net
net = train(net,Xs,Ts,Xi,Ai);
%get new input
% prepare input to get time series for training
[Xsn,Xin,Ain] = preparets(net,Xn);
% get prediction
Yn = net(Xsn,Xin,Ain);