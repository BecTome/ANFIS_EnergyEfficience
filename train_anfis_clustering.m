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


rng(123); % Fix random seed
n = size(X_orig, 1); % Total number of samples
cv = cvpartition(n, 'KFold', N_SPLITS); % cv object -- training, test functions

shuffledIndices = randperm(n); % Shuffle indices
X = X_orig(shuffledIndices, :); % Shuffle X
Y = Y_orig(shuffledIndices, :); % Shuffle Y

% Define the range of hyperparameters to be tuned
cInfR = [0.25, 0.5, 0.75]';
accRt = [0.25, 0.5, 0.75]';
rejRt = [0.15, 0.3, 0.45]';
trainingEpochs = [10 25 50]';

ma=size(cInfR,1);
mb=size(accRt,1);
mc=size(rejRt,1);
md=size(trainingEpochs, 1);

[a, b, c, d]=ndgrid(1:ma,1:mb,1:mc, 1:md);
params = [cInfR(a,:), accRt(b,:), rejRt(c,:), trainingEpochs(d,:)];
params = params(params(:,2) > params(:,3), :); % Ensure acceptance greater than rejections

%%
results = [];
for i = 1:size(params, 1)
   cInfR_i = params(i, 1);
   accRt_i = params(i, 2);
   rejRt_i = params(i, 3);
   trainingEpochs_i = params(i,4);
%    split_i = str2num(params(i, 4));
   runTime = [];
   results_split = [];
   for split = 1:cv.NumTestSets
       fprintf('\nRUN %d/%d --- SPLIT: %d/%d\n', i, size(params, 1), split, N_SPLITS)
        % Training data for this fold
        X_train = X(training(cv, split), :);
        Y_train = Y(training(cv, split), :);
        
        % Testing data for this fold
        X_test = X(test(cv, split), :);
        Y_test = Y(test(cv, split), :); % Set the seed for reproducibility
        
        
        % Hyperparameter tuning
        
        % Generate FIS with given number of membership functions
        opt = genfisOptions('SubtractiveClustering', ...
                            'ClusterInfluenceRange',cInfR_i, ...,
                            'AcceptRatio', accRt_i, ...,
                            'RejectRatio', rejRt_i);
        
        fis = genfis(X_train, Y_train, opt);
        
        % Train the ANFIS model
        % Pass X and Y separately to anfis when using anfisOptions
        
        tic; % Start Timer
        [trainedFis, trainErr, stepsize, testFis, testErr]=anfis([X_train Y_train], ...
                                                                   fis, ...
                                                                   trainingEpochs_i, ...
                                                                   NaN, ...
                                                                   [X_test Y_test]);
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

    newRow = [trainingEpochs_i, {cInfR_i}, {accRt_i}, {rejRt_i},...
                {results_split}, mean(results_split), std(results_split),...
                mean(runTime), std(runTime)];
    results = [results; newRow];
end

% Convert array to table
resultsTable = array2table(results);

% Set column headers
resultsTable.Properties.VariableNames = {'Epochs', 'InfRange', 'AcceptRatio', 'RejectRatio', 'Split',... 
                                         'Mean', 'Std', 'MeanTime', 'StdTime'};

% Sort table by Mean MAE
resultsTable = sortrows(resultsTable, {'Mean', 'MeanTime'});

%%
% Specify the name of the CSV file
filename = 'results_clustering.csv';

% Export the table to CSV
writetable(resultsTable, filename);



