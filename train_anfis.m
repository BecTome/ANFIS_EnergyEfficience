%% Intro
clear;
clc;
close all;

N_SPLITS = 3;

data = readtable("data/ENB2012_data.xlsx");

%% Load your dataset
% Assuming X is an Nx8 matrix (N samples, 8 inputs) and Y is an Nx2 matrix (N samples, 2 outputs)
X_orig = table2array(data(1:100, 1:end-2));
Y_orig = table2array(data(1:100, end-1));

%% Train - Test - Validation split


rng(123); % Fix random seed
n = size(X_orig, 1); % Total number of samples
cv = cvpartition(n, 'KFold', N_SPLITS); % cv object -- training, test functions

% shuffledIndices = randperm(n); % Shuffle indices
X = X_orig(shuffledIndices, :); % Shuffle X
Y = Y_orig(shuffledIndices, :); % Shuffle Y

% Define the range of hyperparameters to be tuned
numMFs = 2;  % Number of Membership Functions (example values)
typeMF = ["gbellmf", "gaussmf", "trimf", "trapmf"];
trainingEpochs = [10 20 30];  % Number of Training Epochs (example values)
outputMF = ["linear", "constant"]; % Sugeno Function

results = [];
% Loop over the range of hyperparameters
for i = 1:length(typeMF)
    fprintf("\nType of MF: %", typeMF(i))
    for j = 1:length(trainingEpochs)
        results_split = [];
        for split = 1:cv.NumTestSets
            fprintf('\nSPLIT: %d', split)
            % Training data for this fold
            X_train = X_orig(training(cv, split), :);
            Y_train = Y_orig(training(cv, split), :);
        
            % Testing data for this fold
            X_test = X_orig(test(cv, split), :);
            Y_test = Y_orig(test(cv, split), :); % Set the seed for reproducibility
        
        
            % Hyperparameter tuning
    
            % Generate FIS with given number of membership functions
            opt = genfisOptions('GridPartition', ...
                                'NumMembershipFunctions',numMFs, ...,
                                'InputMembershipFunctionType', typeMF(i));
    
            fis = genfis(X_train, Y_train, opt);
      
            % Train the ANFIS model
            % Pass X and Y separately to anfis when using anfisOptions
            [trainedFis, trainErr, stepsize, testFis, testErr]=anfis([X_train Y_train], ...
                                                                       fis, ...
                                                                       trainingEpochs(j), ...
                                                                       NaN, ...
                                                                       [X_test Y_test]);
            % Evaluate the performance
            validErr = testErr(end);
            Y_pred = evalfis(testFis, X_test);

            validErr = mean(abs(Y_test - Y_pred));

            % Store the results
            %end
            results_split = [results_split validErr];
        end
        results = [results; trainingEpochs(j) typeMF(i) results_split mean(results_split) std(results_split)];
    end
end



