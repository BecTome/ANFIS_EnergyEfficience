%% Intro
clear;
clc;    
close all;

N_SPLITS = 3;

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
typeMF = ["gbellmf", "gaussmf", "trimf", "trapmf"]';
trainingEpochs = [25 50 75 100]';  % Number of Training Epochs (example values)
outputMF = ["linear", "constant"]'; % Sugeno Function
splits = [1:N_SPLITS]';

ma=size(typeMF,1);
mb=size(trainingEpochs,1);
mc=size(outputMF,1);

[a, b, c]=ndgrid(1:ma,1:mb,1:mc); %, 1:msplit);
params = [typeMF(a,:), trainingEpochs(b,:), outputMF(c,:)];

results = [];
parfor i = 1:size(params, 1)
   fprintf("\nRUN %d of %d", i, size(params, 1))
   typeMF_i = params(i, 1);
   trainingEpochs_i = str2double(params(i, 2));
   outputMF_i = params(i, 3);
%    split_i = str2num(params(i, 4));
   runTime = [];
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
                            'InputMembershipFunctionType', typeMF_i, ...,
                            'OutputMembershipFunctionType', outputMF_i);
        
        fis = genfis(X_train, Y_train, opt);

        
        % Train the ANFIS model
        % Pass X and Y separately to anfis when using anfisOptions
        
        tic; % Start Timer
        [trainedFis, trainErr, stepsize, testFis, testErr]=anfis([X_train Y_train], ...
                                                                   fis, ...
                                                                   trainingEpochs_i, ...
                                                                   NaN, ...
                                                                   [X_test Y_test] ...
                                                                   );
        trainingTime = toc;
        runTime = [runTime trainingTime];
        
        % Evaluate the performance
        validErr = testErr(end);
        Y_pred = evalfis(testFis, X_test);
        
        validErr = mean(abs(Y_test - Y_pred));
        
        % Store the results
        %end
        results_split = [results_split validErr];
   end

    newRow = [trainingEpochs_i, {typeMF_i}, {outputMF_i},...
                {results_split}, mean(results_split), std(results_split),...
                mean(runTime), std(runTime)];
    results = [results; newRow];
end

% Convert array to table
resultsTable = array2table(results);

% Set column headers
resultsTable.Properties.VariableNames = {'Epochs', 'TypeMF', 'OutputMF', 'Split',... 
                                         'Mean', 'Std', 'MeanTime', 'StdTime'};
%%
% Specify the name of the CSV file
filename = 'results_2.csv';

% Export the table to CSV
writetable(resultsTable, filename);



