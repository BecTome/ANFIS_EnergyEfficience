
clear;
clc;
close all;

N_SPLITS = 10;

data = readtable("data/ENB2012_data.xlsx");


% Assuming X is an Nx8 matrix (N samples, 8 inputs) and Y is an Nx2 matrix (N samples, 2 outputs)
X_orig = table2array(data(:, [1, 7]));
Y_orig = table2array(data(:, end-1));

% rng(123); % Fix random seed

% Define the range of hyperparameters to be tuned
numMFs = 5;  % Number of Membership Functions (example values)
typeMF = "gaussmf";
trainingEpochs = 100;  % Number of Training Epochs (example values)
outputMF = "linear"; % Sugeno Function

results = [];
errors_train = [];
errors_test = [];

NRUNS = 10;

% Loop over the range of hyperparameters
parfor i = 1:NRUNS
    n = size(X_orig, 1); % Total number of samples

    shuffledIndices = randperm(n); % Shuffle indices
    X = X_orig(shuffledIndices, :); % Shuffle X
    Y = Y_orig(shuffledIndices, :); % Shuffle Y

    cv = cvpartition(n, 'KFold', N_SPLITS); % cv object -- training, test functions

    results_split = [];
    runTime = [];
    trainErrors = []; % Almacenar errores de entrenamiento
    testErrors = []; % Almacenar errores de prueba
    for split = 1:cv.NumTestSets
        fprintf('\nSPLIT: %d', split)
        % Training data for this fold
        X_train = X(training(cv, split), :);
        Y_train = Y(training(cv, split), :);
    
        % Testing data for this fold
        X_test = X(test(cv, split), :);
        Y_test = Y(test(cv, split), :); % Set the seed for reproducibility
    
        % Generate FIS with given number of membership functions
        opt = genfisOptions('GridPartition', ...
                            'NumMembershipFunctions',numMFs, ...,
                            'InputMembershipFunctionType', typeMF);

        fis = genfis(X_train, Y_train, opt);
  
        tic; % Start Timer
        % Train the ANFIS model
        % Pass X and Y separately to anfis when using anfisOptions
        [trainedFis, trainErr, stepsize, testFis, testErr]=anfis([X_train Y_train], ...
                                                                   fis, ...
                                                                   trainingEpochs, ...
                                                                   NaN, ...
                                                                   [X_test Y_test]);
        trainingTime = toc;
        runTime = [runTime trainingTime];

        % Acumular errores
        trainErrors = [trainErrors; trainErr'];
        testErrors = [testErrors; testErr'];

        % Evaluate the performance
        validErr = testErr(end);
        Y_pred = evalfis(testFis, X_test);

        validErr = mean(abs(Y_test - Y_pred));

        % Store the results
        %end
        results_split = [results_split validErr];
    end
    errors_train = [errors_train; mean(trainErrors, 1)];
    errors_test = [errors_test; mean(testErrors, 1)];
    newRow = [i, trainingEpochs, numMFs, typeMF, outputMF,...
                        results_split, mean(results_split), std(results_split),...
                        mean(runTime), std(runTime)];
    results = [results; newRow];
end

% fprintf("Elapsed Time: %f",toc);
% Convert array to table
resultsTable = array2table(results);

% Set column headers
resultsTable.Properties.VariableNames = {'RUN', 'Epochs', 'numMFs', 'TypeMF', 'OutputMF',...
                                        'Split1', 'Split2', 'Split3', 'Split4', 'Split5',...
                                        'Split6', 'Split7', 'Split8','Split9','Split10',...
                                         'Mean', 'Std', 'MeanTime', 'StdTime'};

%% dffd

% Specify the name of the CSV file
filename = '2var_results_best.csv';

% Export the table to CSV
writetable(resultsTable, filename);

best_results = [mean(str2double(resultsTable.Mean)), std(str2double(resultsTable.Mean))];
table_best_results = array2table(best_results);

table_best_results.Properties.VariableNames = {'Mean', 'Std'};

% Specify the name of the CSV file
filename = '2var_results_best_mean_std.csv';

% Export the table to CSV
writetable(table_best_results, filename);

% Plot error curves
fig = figure;

% Left Y-axis for training error
yyaxis left;
plot(errors_train(end,:), 'b-'); % Plot the last training errors
ylabel('Training Error');

% Right Y-axis for test error
yyaxis right;
plot(errors_test(end,:), 'r-'); % Plot the last test errors
ylabel('Test Error');

xlabel('Epochs');
legend('Training Error', 'Test Error');
title('Training and Test Error Curves');

% Save the figure
saveas(fig, '2var_rmse_best_params.png');

% Plot curves on same Y-axis
fig2 = figure;
plot(errors_train(end,:), 'b-');
hold on;
plot(errors_test(end,:), 'r-');
xlabel('Epochs');
ylabel('Error');
legend('Training Error', 'Test Error');
title('Training and Test Error Curves');

saveas(fig2,'2var_rmse_best_params_same_axis.png')



