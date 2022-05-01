%% PARAMETERS TO TWEAK
% Pre-processing
do_standardize = false;
do_mean_filtering = false;
lf = [8,14];
hf = [13,30];
filtfilt_order = 4;
start_sample = 512;

% Feature engineering
do_log = true;
norm_var = false;

% CSP calculation
csp_corr = false;
csp_diff = false;
filters = 8;

% ERP difference estimation
do_ERPs = false;
erp_filters = 4;

% Boradband PCA (alpha + beta)
do_PCA = false;
n_pcs = 10;

% Do non-stationarity analysis
do_nonsta = true;
block_size = fs*3;
window_step = fs/4;





%%
filenames = ...
{'P1_pre_'
 %'P1_post_'
 %'P2_pre_'
 %'P2_post_'
 %'P3_pre_'
 %'P3_post_'
 };

close all

ACCURACIES = {};


for filename_index = 1:length(filenames)

[Session_Data,fs] = Data_Sorting([filenames{filename_index} 'training.mat']);

[X_left, X_right] = Pre_Process(Session_Data, lf, hf, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering , 0, do_nonsta, block_size, window_step);

%% Calculate and project CSP

V = Get_CSP(X_left,X_right,filters,csp_corr,csp_diff);

[X_left_csp, X_right_csp] = Project_CSP(X_left,X_right,V);



% Wrap trials
if do_nonsta
    X_left_csp = reshape(X_left_csp,block_size,[],filters*length(hf));
    X_right_csp = reshape(X_right_csp,block_size,[],filters*length(hf));
else
    X_left_csp = reshape(X_left_csp,2048-start_sample+1,[],filters*length(hf));
    X_right_csp = reshape(X_right_csp,2048-start_sample+1,[],filters*length(hf));
end


% Feature engineering
if norm_var
    den_left = squeeze(sum(var(X_left_csp),3));
    den_right = squeeze(sum(var(X_right_csp),3));
    X_left_feature = (squeeze(var(X_left_csp)) ./ repmat(den_left',1,filters*length(hf)));
    X_right_feature = (squeeze(var(X_right_csp)) ./ repmat(den_right',1,filters*length(hf)));
else
    X_left_feature = squeeze(var(X_left_csp));
    X_right_feature = squeeze(var(X_right_csp));
end

if do_log
   X_left_feature = log(X_left_feature);
   X_right_feature = log(X_right_feature);
end


%% Add ERP processing to the model
if do_ERPs
    [ERP_left, ERP_right] = Pre_Process(Session_Data, 1, 4, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering, 1, do_nonsta, block_size, window_step);
    V_ERP = Get_CSP(ERP_left,ERP_right,erp_filters,0,0);
    [ERP_left_csp, ERP_right_csp] = Project_CSP(ERP_left,ERP_right,V_ERP);
    
    ERP_left_csp = reshape(ERP_left_csp,256+1,[],erp_filters);
    ERP_right_csp = reshape(ERP_right_csp,256+1,[],erp_filters);
    
    den_left_erp = squeeze(sum(var(ERP_left_csp),3));
    den_right_erp = squeeze(sum(var(ERP_right_csp),3));
    
    ERP_left_csp_feature = squeeze(var(ERP_left_csp))./ repmat(den_left_erp',1,erp_filters);
    ERP_right_csp_feature = squeeze(var(ERP_right_csp)) ./ repmat(den_right_erp',1,erp_filters);
    
    if do_log
        ERP_left_csp_feature = log(ERP_left_csp_feature);
        ERP_right_csp_feature = log(ERP_right_csp_feature);
    end
    X_left_feature = [X_left_feature,ERP_left_csp_feature];
    X_right_feature = [X_right_feature,ERP_right_csp_feature];
end


%% Add variance of the first n PC's of the model
if do_PCA
    [PCA_left, PCA_right] = Pre_Process(Session_Data, 8, 30, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering, 0 ,0);
    [ descriptives, lambdas, Aij, Rij, Lij, Yij, explained, var_weights, relevant_vars, relevant_pcs] = pca_report([PCA_left;PCA_right], 10, 1 );
    PCA_left = PCA_left * Aij(:,1:n_pcs);
    PCA_right = PCA_right * Aij(:,1:n_pcs);
    PCA_left = reshape(PCA_left,2048-start_sample+1,[],n_pcs);
    PCA_right = reshape(PCA_right,2048-start_sample+1,[],n_pcs);
    PCA_left_ = squeeze(var(PCA_left))./repmat(sum(squeeze(var([PCA_left;PCA_right]))),40,1);
    PCA_right_ = squeeze(var(PCA_right))./repmat(sum(squeeze(var([PCA_left;PCA_right]))),40,1);
    if do_log
        PCA_left_ = log(PCA_left_);
        PCA_right_ = log(PCA_right_);
    end
    X_left_feature = [X_left_feature,PCA_left_];
    X_right_feature = [X_right_feature,PCA_right_];
end


%% Calculate LDA parameters
lambda = 0.05;

mu2 = mean(X_left_feature)';
mu1 = mean(X_right_feature)';
E2 = cov(X_left_feature);
E1 = cov(X_right_feature);

I1 = eye(length(E1));
I2 = eye(length(E2));

E1_tilda = (1 - lambda) * E1 + (lambda * I1);
E2_tilda = (1 - lambda) * E2 + (lambda * I2);

theta = (((E1_tilda + E2_tilda)^-1) * (mu2 - mu1)) ;
b = -theta'*(mu1 + mu2)/2;

%% Test

Session_Data = Data_Sorting([filenames{filename_index} 'test.mat']);
[X_left, X_right] = Pre_Process(Session_Data, lf, hf, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering,0, do_nonsta, block_size, window_step);
[X_left_csp, X_right_csp] = Project_CSP(X_left,X_right,V);

% Wrap trials
if do_nonsta
    X_left_csp = reshape(X_left_csp,block_size,[],filters*length(hf));
    X_right_csp = reshape(X_right_csp,block_size,[],filters*length(hf));
else
    X_left_csp = reshape(X_left_csp,2048-start_sample+1,[],filters*length(hf));
    X_right_csp = reshape(X_right_csp,2048-start_sample+1,[],filters*length(hf));
end

if norm_var
    den_left = squeeze(sum(var(X_left_csp),3));
    den_right = squeeze(sum(var(X_right_csp),3));
    X_left_feature = (squeeze(var(X_left_csp)) ./ repmat(den_left',1,filters*length(hf)));
    X_right_feature = (squeeze(var(X_right_csp)) ./ repmat(den_right',1,filters*length(hf)));
else
    X_left_feature = squeeze(var(X_left_csp));
    X_right_feature = squeeze(var(X_right_csp));
end

if do_log
   X_left_feature = log(X_left_feature);
   X_right_feature = log(X_right_feature);
end

if do_ERPs
    [ERP_left, ERP_right] = Pre_Process(Session_Data, 1, 4, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering, 1);
    [ERP_left_csp, ERP_right_csp] = Project_CSP(ERP_left,ERP_right,V_ERP);
    ERP_left_csp = reshape(ERP_left_csp,256+1,[],erp_filters);
    ERP_right_csp = reshape(ERP_right_csp,256+1,[],erp_filters);
    ERP_left_csp_feature = squeeze(var(ERP_left_csp))./ repmat(den_left_erp',1,erp_filters);
    ERP_right_csp_feature = squeeze(var(ERP_right_csp)) ./ repmat(den_right_erp',1,erp_filters);
    if do_log
        ERP_left_csp_feature = log(ERP_left_csp_feature);
        ERP_right_csp_feature = log(ERP_right_csp_feature);
    end
    X_left_feature = [X_left_feature,ERP_left_csp_feature];
    X_right_feature = [X_right_feature,ERP_right_csp_feature];
end

if do_PCA
    [PCA_left, PCA_right] = Pre_Process(Session_Data, 8, 30, filtfilt_order, fs, start_sample, do_standardize, do_mean_filtering, 0);
    PCA_left = PCA_left * Aij(:,1:n_pcs);
    PCA_right = PCA_right * Aij(:,1:n_pcs);
    PCA_left = reshape(PCA_left,2048-start_sample+1,[],n_pcs);
    PCA_right = reshape(PCA_right,2048-start_sample+1,[],n_pcs);
    PCA_left_ = squeeze(var(PCA_left))./repmat(sum(squeeze(var([PCA_left;PCA_right]))),40,1);
    PCA_right_ = squeeze(var(PCA_right))./repmat(sum(squeeze(var([PCA_left;PCA_right]))),40,1);
    if do_log
        PCA_left_ = log(PCA_left_);
        PCA_right_ = log(PCA_right_);
    end
    X_left_feature = [X_left_feature,PCA_left_];
    X_right_feature = [X_right_feature,PCA_right_];
end







test_set = [X_left_feature;X_right_feature];

if do_nonsta
    
    total_size = size(test_set,1);
    blocks_per_trial = total_size / 80;
    
    y_extended = sign(theta' * test_set' + b)';
    
    trials = 1:blocks_per_trial:total_size;
    
    y = [];
    for t = trials
        y = [y;sign(mean(y_extended(t:t+blocks_per_trial-1)))];
    end
    
    
    
else
    y = sign(theta' * test_set' + b)';
end

   
    test_labels = [ones(40,1);-1*ones(40,1)];
    acc = mean(y == test_labels);

    ACCURACIES = [ACCURACIES;{filenames{filename_index},acc}];



end





