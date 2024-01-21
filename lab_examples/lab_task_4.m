clc;
clear;
close;

% Generate 51 input-output pairs between x ϵ [-10, 10], and choose training
% and checking data sets:

numPts=51;
x=linspace(-10,10,numPts)';
y=-2*x-x.^2;
data=[x y];
trndata=data(1:2:numPts,:);
chkdata=data(2:2:numPts,:);

% Take a look to the training output:
plot(trndata(:,1),trndata(:,2),'*r')

% and the cheking output:
hold on
plot(chkdata(:,1),chkdata(:,2),'*b')

% Set the number and type of membership functions:
numMFs=5;
mfType='gbellmf';

opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = numMFs;
opt.InputMembershipFunctionType = mfType;

% Generate the FIS-matrix and execute the ANFIS-training by 40 rounds. 
% genfis generates a Sugeno-type FIS (Fuzzy Inference System ) structure 
% from data using grid partition and uses it as initial condition for anfis 
% training. anfis is the training routine for Sugeno-type FIS. It uses a 
% hybrid learning (least-squares + gradient descent) algorithm to identify 
% the parameters of Sugeno-type FIS.

fismat=genfis(trndata(:, 1), trndata(:, 2), opt);
numEpochs=1;
[fismat1,trnErr,stepsize,fismat2,chkErr]=anfis(trndata,fismat,numEpochs,NaN,chkdata);


%Compare training and checking data to the fuzzy approximation. evalfis 
% performs fuzzy inferencecalculations.
anfis_y=evalfis(fismat1,x(:,1));
plot(trndata(:,1),trndata(:,2),'o', ...
     chkdata(:,1),chkdata(:,2),'x', ...
     x,anfis_y,'-')

% Draw also original function to the same picture so it is possible to 
% compare function and fuzzy approximation:
hold on;
plot(x,y)
writeFIS(fismat1,'lab_examples/fismat1');