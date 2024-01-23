%% Intro
clear;
clc;    
close all;

data = readtable("data/ENB2012_data.xlsx");

%% Load your dataset
% Assuming X is an Nx8 matrix (N samples, 8 inputs) and Y is an Nx2 matrix (N samples, 2 outputs)
X_orig = table2array(data(:, 1:end-2));
Y_orig = table2array(data(:, end-1));

%% Train - Test - Validation split
n = size(X_orig, 1); % Total number of samples
shuffledIndices = randperm(n); % Shuffle indices
X = X_orig(shuffledIndices, :); % Shuffle X
Y = Y_orig(shuffledIndices, :); % Shuffle Y

% Define the range of hyperparameters to be tuned
numMFs = 2;  % Number of Membership Functions (example values)
typeMF = "gbellmf";
trainingEpochs = 50;  % Number of Training Epochs (example values)
outputMF = "constant"; % Sugeno Function


X_train = X_orig(1:500, :);
Y_train = Y_orig(1:500, :);

% Testing data for this fold
X_test = X_orig(501:end, :);
Y_test = Y_orig(501:end, :); % Set the seed for reproducibility

% Generate FIS with given number of membership functions
opt = genfisOptions('GridPartition', ...
                    'NumMembershipFunctions',numMFs, ...,
                    'InputMembershipFunctionType', typeMF, ...,
                    'OutputMembershipFunctionType', outputMF);

fis = genfis(X_train, Y_train, opt);

        
% Train the ANFIS model
% Pass X and Y separately to anfis when using anfisOptions

tic; % Start Timer
[trainedFis, trainErr, stepsize, testFis, testErr]=anfis([X_train Y_train], ...
                                                           fis, ...
                                                           trainingEpochs, ...
                                                           NaN, ...
                                                           [X_test Y_test] ...
                                                           );
trainingTime = toc;

% Plot curves on same Y-axis
fig2 = figure;
plot(trainErr, 'b-');
hold on;
plot(testErr, 'r-');
xlabel('Epochs');
ylabel('Error');
legend('Training Error', 'Test Error');
title('Training and Test Error Curves');

saveas(fig2,'onerun.png')
