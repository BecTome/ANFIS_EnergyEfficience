%% Intro
clear;
clc;
close all;


data = readtable("data/ENB2012_data.xlsx");

% Load your dataset
% Assuming X is an Nx8 matrix (N samples, 8 inputs) and Y is an Nx2 matrix (N samples, 2 outputs)

X = table2array(data(1:10, 1:3));
Y = table2array(data(1:10, end-1:end));

% Define the range of hyperparameters to be tuned
numMFs = [2, 4];  % Number of Membership Functions (example values)
trainingEpochs = [3, 3];  % Number of Training Epochs (example values)

% Initialize an array to store the results
results = [];

% Loop over the range of hyperparameters
for i = 1:length(numMFs)
    disp(i)
    for j = 1:length(trainingEpochs)
        % Generate FIS with given number of membership functions
        opt = genfisOptions('GridPartition', ...
                            'NumMembershipFunctions',numMFs(i), ...
                            'InputMembershipFunctionType', 'gbellmf');
        fis = genfis(X, Y, opt);

        % Setup training options
        %anfisOptions = anfisOptions( 'EpochNumber', trainingEpochs(j));
        
        % Train the ANFIS model
        % Pass X and Y separately to anfis when using anfisOptions
        [trainedFis, trainError] = anfis([X, Y], fis, trainingEpochs(j));

        % Evaluate the performance
        finalError = trainError(end);

        % Store the results
        results = [results; numMFs(i), trainingEpochs(j), finalError];
    end
end

% Display the results
disp('NumMFs Epochs FinalError');
disp(results);

% Select the best hyperparameters based on the lowest error
[~, bestIdx] = min(results(:, 3));
bestNumMFs = results(bestIdx, 1);
bestEpochs = results(bestIdx, 2);

fprintf('Best NumMFs: %d, Best Epochs: %d\n', bestNumMFs, bestEpochs);
