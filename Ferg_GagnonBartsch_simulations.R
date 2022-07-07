##############################################################################
##############################################################################
#### Simulations for 'Unbiased Survey Estimation with Population Auxiliary Variables'
#### by Ferg, Gagnon-Bartsch
##############################################################################
##############################################################################

library(ggplot2)
library(tidyverse)
library(randomForest)
library(e1071)


##############################################################################
#### Slightly Non-Linear Relationship
##############################################################################

set.seed(123)

N_snl = 1000
X_snl = rnorm(N_snl)
Y_snl = ifelse(X_snl>-0.5, 2*X_snl+.5, 0) + rnorm(N_snl)
data_snl = data.frame('X'=X_snl, 'Y'=Y_snl)

ggplot(data_snl, aes(x=X, y=Y)) + geom_point() + theme_bw()

ns_snl = c(10, 50, 100)
nsims_snl = 10000

ests_snl = data.frame()
for(n in ns_snl){
  gammas = matrix(rep(1/n, (N_snl-n)*n), nrow=N_snl-n, ncol=n)
  pis = rep(n/N_snl, n)
  pijs = matrix(n*(n-1)/(N_snl*(N_snl-1)), nrow=n, ncol=n)
  for(i in 1:nsims_snl){
    # get i-th sample
    samp = sample(1:N_snl, n)
    x_samp = X_snl[samp]
    y_samp = Y_snl[samp]
    x_other = X_snl[-samp]
    # using our estimator with OLS
    new_est = newEstimator(x_samp, y_samp, x_other, pis, pijs, pred.fun='lm', gammas)
    # using model-assisted with OLS
    ma_OLS_est = modelAssisted_OLS(y_samp, x_samp, sum(x_other), N_snl, pis, pijs)
    # using sample mean
    samp_mean_est = mean(y_samp)
    samp_mean_var = var(y_samp)/n
    # save results
    temp = data.frame('n' = n, 'muhat' = new_est$muHat, 'varEst' = new_est$varEst,
                      'muhat_MA' = ma_OLS_est$muhat, 'varEst_MA' = ma_OLS_est$varEst,
                      'muhat_SM' = samp_mean_est, 'varEst_SM' = samp_mean_var)
    ests_snl = rbind(ests_snl, temp)
  }
}

summaryStats(ests_snl, 'muhat', 'varEst', mean(Y_snl))
summaryStats(ests_snl, 'muhat_MA', 'varEst_MA', mean(Y_snl))
summaryStats(ests_snl, 'muhat_SM', 'varEst_SM', mean(Y_snl))


##############################################################################
#### Non-Linear Relationship
##############################################################################

set.seed(234)

N_nl = 1000
X_nl = matrix(runif(10*N_nl), ncol=10, nrow=N_nl) %>% data.frame()
Y_nl = .1*exp(4*X_nl[,1]) + 4/(1+exp(-1*(X_nl[,2]-.5)/.05)) + 10*sin(pi*X_nl[,3]*X_nl[,4]) + 3*X_nl[,5] + rnorm(N_nl)

ns_nl = c(25, 50)
nsims_nl = 10000

ests_nl = data.frame()
for(n in ns_nl){
  gammas = matrix(rep(1/n, (N_nl-n)*n), nrow=N_nl-n, ncol=n)
  pis = rep(n/N_nl, n)
  pijs = matrix(n*(n-1)/(N_nl*(N_nl-1)), nrow=n, ncol=n)
  for(i in 1:nsims_nl){
    # get i-th sample
    samp = sample(1:N_nl, n)
    x_samp = X_nl[samp,]
    y_samp = Y_nl[samp]
    x_other = X_nl[-samp,]
    # using our estimator with OLS
    new_est = newEstimator(x_samp, y_samp, x_other, pis, pijs, pred.fun='lm', gammas)
    # using model-assisted with OLS
    ma_OLS_est = modelAssisted_OLS(y_samp, x_samp, colSums(x_other), N_nl, pis, pijs)
    # using our estimator with random forest
    new_est_rf = newEstimator(x_samp, y_samp, x_other, pis, pijs, pred.fun='rf', gammas)
    # using model-assisted with random forest
    ma_RF_est = modelAssisted_ML(y_samp, x_samp, x_other, pis, pijs, method='rf')
    # using our estimator with SVM
    new_est_svm = newEstimator(x_samp, y_samp, x_other, pis, pijs, pred.fun='svm', gammas)
    # using model-assisted with SVM
    ma_svm_est = modelAssisted_ML(y_samp, x_samp, x_other, pis, pijs, method='svm')
    # using sample mean
    samp_mean_est = mean(y_samp)
    samp_mean_var = var(y_samp)/n
    # save results
    temp = data.frame('n' = n, 'muhat_ols' = new_est$muHat, 'varEst_ols' = new_est$varEst,
                      'muhat_MA_ols' = ma_OLS_est$muhat, 'varEst_MA_ols' = ma_OLS_est$varEst,
                      'muhat_rf' = new_est_rf$muHat, 'varEst_rf' = new_est_rf$varEst,
                      'muhat_MA_rf' = ma_RF_est$muhat, 'varEst_MA_rf' = ma_RF_est$varEst,
                      'muhat_svm' = new_est_svm$muHat, 'varEst_svm' = new_est_svm$varEst,
                      'muhat_MA_svm' = ma_svm_est$muhat, 'varEst_MA_svm' = ma_svm_est$varEst,
                      'muhat_SM' = samp_mean_est, 'varEst_SM' = samp_mean_var)
    ests_nl = rbind(ests_nl, temp)
    print(i)
  }
}

summaryStats(ests_nl, 'muhat_ols', 'varEst_ols', mean(Y_nl))
summaryStats(ests_nl, 'muhat_MA_ols', 'varEst_MA_ols', mean(Y_nl))
summaryStats(ests_nl, 'muhat_rf', 'varEst_rf', mean(Y_nl))
summaryStats(ests_nl, 'muhat_MA_rf', 'varEst_MA_rf', mean(Y_nl))
summaryStats(ests_nl, 'muhat_svm', 'varEst_svm', mean(Y_nl))
summaryStats(ests_nl, 'muhat_MA_svm', 'varEst_MA_svm', mean(Y_nl))
summaryStats(ests_nl, 'muhat_SM', 'varEst_SM', mean(Y_nl))


##############################################################################
#### Stratified Sample
##############################################################################

set.seed(345)

N_strat = 1000
X_1 = c(rep(1, .5*N_strat), rep(2, .25*N_strat), rep(3, .25*N_strat)) %>% as.factor()
X_strat = data.frame('s1'=ifelse(X_1==1, 1, 0),
                     's2'=ifelse(X_1==2, 1, 0), 
                     matrix(rnorm(N_strat*3), ncol=3, nrow=N_strat))
Y_strat = ifelse(X_1==1, X_strat[,2]-2*X_strat[,3]+3*X_strat[,4],
               ifelse(X_1==2, 2*X_strat[,2]-2*X_strat[,3]+3*X_strat[,4],
                      X_strat[,2]-3*X_strat[,3]+X_strat[,4])) + rnorm(N_strat)

ns_strat = rep(10, 3)
n_strat = sum(ns_strat)
pis_strat = c(rep(ns_strat[1]/sum(X_1==1), ns_strat[1]), 
              rep(ns_strat[2]/sum(X_1==2), ns_strat[2]), 
              rep(ns_strat[3]/sum(X_1==3), ns_strat[3]))
pijs_strat = matrix(NA, nrow=sum(ns_strat), ncol=sum(ns_strat))
for(i in 1:(nrow(pijs_strat)-1)){
  for(j in (i+1):nrow(pijs_strat)){
    if(X_1[i]==X_1[j]) pij = ns_strat[X_1[i]]*(ns_strat[X_1[i]]-1)/(sum(X_1==X_1[i])*(sum(X_1==X_1[i])-1))
    else pij = ns_strat[X_1[i]]*ns_strat[X_1[j]]/(sum(X_1==X_1[i])*sum(X_1==X_1[j]))
    pijs_strat[i,j] = pij; pijs_strat[j,i] = pij
  }
}

nsims_strat = 10000
ests_strat = data.frame()
for(i in 1:nsims_strat){
  # get i-th sample
  samp = c(sample(which(X_1==1), ns_strat[1], replace=FALSE),
           sample(which(X_1==2), ns_strat[2], replace=FALSE),
           sample(which(X_1==3), ns_strat[3], replace=FALSE))
  x_samp = X_strat[samp,]
  y_samp = Y_strat[samp]
  x_other = X_strat[-samp,]
  # using our new estimator with OLS
  new_est = newEstimator_stratified(x_samp, y_samp, x_other, pis_strat, pijs_strat, pred.fun='lm', x_samp[,1], x_other[,1])
  # using model-assisted with OLS
  ma_OLS_est = modelAssisted_OLS(y_samp, x_samp, colSums(x_other), N_strat, pis_strat, pijs_strat)
  # using our estimator with random forest
  new_est_rf = newEstimator_stratified(x_samp, y_samp, x_other, pis_strat, pijs_strat, pred.fun='rf', x_samp[,1], x_other[,1])
  # using model-assisted with random forest
  ma_RF_est = modelAssisted_ML(y_samp, x_samp, x_other, pis_strat, pijs_strat, method='rf')
  # save results
  temp = data.frame('n' = n_strat, 'muhat' = new_est$muHat, 'varEst' = new_est$varEst,
                    'muhat_MA' = ma_OLS_est$muhat, 'varEst_MA' = ma_OLS_est$varEst,
                    'muhat_rf' = new_est_rf$muHat, 'varEst_rf' = new_est_rf$varEst,
                    'muhat_MA_rf' = ma_RF_est$muhat, 'varEst_MA_rf' = ma_RF_est$varEst)
  ests_strat = rbind(ests_strat, temp)
}

summaryStats(ests_strat, 'muhat', 'varEst', mean(Y_strat))
summaryStats(ests_strat, 'muhat_MA', 'varEst_MA', mean(Y_strat))
summaryStats(ests_strat, 'muhat_rf', 'varEst_rf', mean(Y_strat))
summaryStats(ests_strat, 'muhat_MA_rf', 'varEst_MA_rf', mean(Y_strat))



##############################################################################
#### High-Dimensional Sample
##############################################################################

set.seed(567)

N_hd = 500

X_hd = matrix(rnorm(N_hd*1000), nrow=N_hd, ncol=1000)
Y_hdc = X_hd[,1] + 2*X_hd[,2] + 2*X_hd[,3] + rnorm(N_hd)
Y_hdb = rbinom(N_hd, 1, prob=exp(X_hd[,1] + 2*X_hd[,2] + 2*X_hd[,3])/(1+exp(X_hd[,1] + 2*X_hd[,2] + 2*X_hd[,3])))

n_hd = 50

gammas = matrix(rep(1/n_hd, (N_hd-n_hd)*n_hd), nrow=N_hd-n_hd, ncol=n_hd)
pis = rep(n_hd/N_hd, n_hd)
pijs = matrix(n_hd*(n_hd-1)/(N_hd*(N_hd-1)), nrow=n_hd, ncol=n_hd)

nsims_hd = 400
ests_hd = data.frame()
for(i in 1:nsims_hd){
  samp = sample(1:N_hd, n_hd)
  x_samp = X_hd[samp,] %>% data.frame()
  yc_samp = Y_hdc[samp]
  yb_samp = Y_hdb[samp]
  x_other = X_hd[-samp,] %>% data.frame()
  # continuous 
  ## our estimator with random forest
  new_cont_rf = newEstimator(x_samp, yc_samp, x_other, pis, pijs, pred.fun='rf', gammas)
  ## model-assisted with random forest
  ma_cont_rf = modelAssisted_ML(yc_samp, x_samp, x_other, pis, pijs, method='rf')
  ## our estimator with LASSO
  new_cont_lasso = newEstimator(x_samp, yc_samp, x_other, pis, pijs, pred.fun='lasso', gammas)
  ## model-assisted with LASSO
  ma_cont_lasso = modelAssisted_ML(yc_samp, x_samp, x_other, pis, pijs, method='lasso')
  # binary
  ## our estimator with random forest
  new_bin_rf = newEstimator(x_samp, yb_samp, x_other, pis, pijs, pred.fun='rf', gammas)
  ## model-assisted with random forest
  ma_bin_rf = modelAssisted_ML(yb_samp, x_samp, x_other, pis, pijs, method='rf')
  ## our estimator with logistic regression
  new_bin_logreg = newEstimator(x_samp, yb_samp, x_other, pis, pijs, pred.fun='logreg1', gammas)
  ## model-assisted with logistic regression
  ma_bin_logreg = modelAssisted_ML(yb_samp, x_samp, x_other, pis, pijs, method='logreg1')
  # save results
  temp = data.frame('n'=n_hd, 'muhat_cont_rf' = new_cont_rf$muHat, 'varEst_cont_rf' = new_cont_rf$varEst,
                    'MA_muhat_cont_rf' = ma_cont_rf$muhat, 'MA_varEst_cont_rf' = ma_cont_rf$varEst,
                    'muhat_cont_lasso' = new_cont_lasso$muHat, 'varEst_cont_lasso' = new_cont_lasso$varEst,
                    'MA_muhat_cont_lasso' = ma_cont_lasso$muhat, 'MA_varEst_cont_lasso' = ma_cont_lasso$varEst,
                    'muhat_bin_rf' = new_bin_rf$muHat, 'varEst_bin_rf' = new_bin_rf$varEst,
                    'MA_muhat_bin_rf' = ma_bin_rf$muhat, 'MA_varEst_bin_rf' = ma_bin_rf$varEst,
                    'muhat_bin_logreg' = new_bin_logreg$muHat, 'varEst_bin_logreg' = new_bin_logreg$varEst,
                    'MA_muhat_bin_logreg' = ma_bin_logreg$muhat, 'MA_varEst_bin_logreg' = ma_bin_logreg$varEst)
  ests_hd = rbind(temp, ests_hd)
}

summaryStats(ests_hd, 'muhat_cont_rf', 'varEst_cont_rf', mean(Y_hdc))
summaryStats(ests_hd, 'MA_muhat_cont_rf', 'MA_varEst_cont_rf', mean(Y_hdc))
summaryStats(ests_hd, 'muhat_cont_lasso', 'varEst_cont_lasso', mean(Y_hdc))
summaryStats(ests_hd, 'MA_muhat_cont_lasso', 'MA_varEst_cont_lasso', mean(Y_hdc))
summaryStats(ests_hd, 'muhat_bin_rf', 'varEst_bin_rf', mean(Y_hdb))
summaryStats(ests_hd, 'MA_muhat_bin_rf', 'MA_varEst_bin_rf', mean(Y_hdb))
summaryStats(ests_hd, 'muhat_bin_logreg', 'varEst_bin_logreg', mean(Y_hdb))
summaryStats(ests_hd, 'MA_muhat_bin_logreg', 'MA_varEst_bin_logreg', mean(Y_hdb))


