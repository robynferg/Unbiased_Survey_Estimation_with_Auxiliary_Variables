library(randomForest)
library(e1071)
library(xgboost)
library(glmnet)

newEstimator = function(Xsample, Ysample, Xother, pis, pijs, pred.fun = 'lm', gammas){
  # Xsample = X variable(s) for sample -- data frame
  # Ysample = outcome for sample -- vector
  # Xother = X variable(s) for rest of the population w/ unknown Ys -- data frame with same variables as Xsample
  # pis = P(i in S) for all i in S
  # pijs = P(i,j in S) for i,j in S -- nxn matrix
  # pred.fun = function for predicting Y from X using sample:
  # 'lm' for OLS
  # 'rf' for random forest
  # 'mean' for sample mean
  # gammas: gamma_i(S\j) for each i not in S and j in S
  
  # convert Xsample and Xother to data frames if vectors
  if(class(Xsample)[1]=='numeric'){
    Xsample = data.frame('x'=matrix(Xsample, ncol=1))
  }
  if(class(Xother)[1]=='numeric'){
    Xother = data.frame('x'=matrix(Xother, ncol=1))
  }
  
  n = nrow(Xsample) # sample size
  N = nrow(Xsample) + nrow(Xother) # population size
  
  # M = data frame with columns: 
  # xlink: index
  # Y: response (if observed)
  # s: 0/1 indicator for being in the sample
  # f: predicted leave-one-out response = f(x_i; s\i)
  M = data.frame('xlink' = c(rownames(Xsample), rep(NA, N-n)),
                 'Y' = c(Ysample, rep(NA, N-n)),
                 's' = c(rep(1, n), rep(0, N-n)),
                 'f' = rep(NA, N))
  # G = data frame of predicted responses for each left out observation
  G = data.frame('Mrow' = rownames(M)[(n+1):N])
  
  # OLS for prediction function
  if(pred.fun=='lm'){
    OLS = lm(Ysample~., data=cbind(Ysample, Xsample))
    # beta-hat for entire sample
    betahat = OLS$coefficients 
    # beta-hat when dropping each individual observation
    betahatDropped = -1*lm.influence(OLS, do.coef=TRUE)$coefficients + matrix(rep(betahat, each=n), nrow=n)
    # f_i's
    M$f = c(rowSums(cbind(rep(1, n), Xsample) * betahatDropped), rep(NA, N-n))
    # G matrix: estimates when leaving out individual observations
    G = cbind(G, as.matrix(cbind(rep(1, N-n), Xother)) %*% t(betahatDropped))
    names(G) = c(names(G)[1], paste0('leftOutObs', names(G)[2:length(names(G))]))
  }
  
  # random forest for prediction function
  # using out-of-bag predictions/trees for in-sample, entire random forest for out-of-sample
  else if(pred.fun=='rf'){
    # random forest for entire sample
    f = randomForest(Ysample~., data=cbind(Ysample, Xsample))
    # predictions for sample -- out-of-bag by default
    M$f = c(f$predicted, rep(NA, N-n))
    # predictions for out-of-sample predictions
    # ignore exact predictions when sample obs. j is out-of-bag since estimates average out to be entire random forest
    g = predict(f, newdata=Xother)
    # G matrix: make entire row the prediction
    G = cbind(G, matrix(rep(g, n), nrow=N-n, ncol=n))
    names(G) = c(names(G)[1], paste0('leftOutObs', 1:n))
  }
  
  # support vector machine
  else if(pred.fun=='svm'){
    for(i in 1:n){
      # leave out observation i and train model
      svm_noi = svm(y=Ysample[-i], x=Xsample[-i,])
      # predict y_i when trained on s\i
      M$f[i] = predict(svm_noi, newdata = Xsample[i,])
      # predict y_j for j not in s
      G[[paste0('x', i)]] = predict(svm_noi, newdata=Xother)
    }
  }
  
  # xgboost
  else if(pred.fun=='xgboost'){
    Xother_xgb = xgb.DMatrix(data=data.matrix(Xother))
    for(i in 1:n){
      # leave out observation i and train model
      sample_xgb_noi = xgb.DMatrix(data=data.matrix(Xsample[-i,]), label=Ysample[-i])
      xgb_noi = xgboost(data=sample_xgb_noi, objective="reg:squarederror", nrounds=50, verbose=0)
      # predict y_i when trained on s\i
      M$f[i] = predict(xgb_noi, newdata=xgb.DMatrix(data=data.matrix(Xsample[i,])))
      # predict y_j for j not in s
      G[[paste0('x', i)]] = predict(xgb_noi, newdata=Xother_xgb)
    }
  }
  
  # LASSO
  else if(pred.fun=='lasso'){
    cv_model = cv.glmnet(data.matrix(Xsample), Ysample, alpha=1)
    lambda_min = cv_model$lambda.min
    for(i in 1:n){
      # leave out observation i and train model
      lasso_noi = glmnet(data.matrix(Xsample[-i,]), Ysample[-i], alpha=1, lambda=lambda_min)
      # predict y_i when trained on s\i
      M$f[i] = predict(lasso_noi, newx=data.matrix(Xsample[i,]))
      # predict y_j for j not in s
      G[[paste('x', i)]] = predict(lasso_noi, newx=data.matrix(Xother))
    }
  }
  
  # L-1 penalty logistic regression
  else if(pred.fun=='logreg1'){
    cv_model = cv.glmnet(data.matrix(Xsample), Ysample, alpha=1, family='binomial')
    lambda_min = cv_model$lambda.min
    for(i in 1:n){
      # leave out observation i and train model
      lasso_noi = glmnet(data.matrix(Xsample[-i,]), Ysample[-i], alpha=1, lambda=lambda_min, family='binomial')
      # predict y_i when trained on s\i
      M$f[i] = predict(lasso_noi, newx=data.matrix(Xsample[i,]), type='response')
      # predict y_j for j not in s
      G[[paste('x', i)]] = predict(lasso_noi, newx=data.matrix(Xother), type='response')
    }
  }
  
  # sample mean for prediction function
  else if(pred.fun=='mean'){
    M$f = c(n/(n-1)*mean(Ysample) - 1/(n-1)*Ysample, rep(NA, N-n))
    G = cbind(G, matrix(rep(n/(n-1)*mean(Ysample) - 1/(n-1)*Ysample, each=N-n), nrow=N-n, ncol=n))
    names(G) = c(names(G)[1], paste0('leftOutObs', 1:n))
  }
  
  # g_i (sum of f's * gammas) for observations not in s
  M$g = c(rep(NA, n), rowSums(G[,-which(names(G)=='Mrow')]*gammas))
  
  # h_i: f_i if i in sample, g_i if i not in sample
  M$h = ifelse(M$s==1, M$f, M$g)
  
  # Y-hat formula
  M$pis = c(pis, rep(NA, N-n))
  M$Yhat = ifelse(M$s==1, M$h + 1/M$pis * (M$Y-M$h), M$h)
  
  muHat = mean(M$Yhat)
  
  # variance estimation
  tempMatrix = as.matrix(na.omit((M$Y-M$f))) %*% t(as.matrix(na.omit(M$Y-M$f)))
  varEst = 1/N^2 * with(M[M$s==1,], sum((1-pis)/pis^2*(Y-f)^2)) + 1/N^2 * sum((pijs*tempMatrix)[upper.tri(tempMatrix)])
  
  return(list('muHat' = muHat, 'M' = M, 'G' = G, 'varEst'=varEst))
}


newEstimator_stratified = function(Xsample, Ysample, Xother, pis, pijs, pred.fun = 'lm', strats_samp, strats_other){
  # Xsample = X variable(s) for sample -- data frame
  # Ysample = outcome for sample -- vector
  # Xother = X variable(s) for rest of the population w/ unknown Ys -- data frame with same variables as Xsample
  # pis = P(i in S) for all i in S
  # pijs = P(i,j in S) for i,j in S -- nxn matrix
  # pred.fun = function for predicting Y from X using sample:
  # 'lm' for OLS
  # 'rf' for random forest
  # 'mean' for sample mean
  # strats_samp: vector indicating which strata each sample observation belongs to
  # strats_other: vector indicating which strat each observation not in the sample belongs to
  
  # convert Xsample and Xother to data frames if vectors
  if(class(Xsample)[1]=='numeric'){
    Xsample = data.frame('x'=matrix(Xsample, ncol=1))
  }
  if(class(Xother)[1]=='numeric'){
    Xother = data.frame('x'=matrix(Xother, ncol=1))
  }
  
  n = nrow(Xsample) # sample size
  N = nrow(Xsample) + nrow(Xother) # population size
  
  # M = data frame with columns: 
  # xlink: index
  # Y: response (if observed)
  # s: 0/1 indicator for being in the sample
  # f: predicted leave-one-out response = f(x_i; s\i)
  M = data.frame('xlink' = c(rownames(Xsample), rep(NA, N-n)),
                 'Y' = c(Ysample, rep(NA, N-n)),
                 's' = c(rep(1, n), rep(0, N-n)),
                 'f' = rep(NA, N))
  # G = data frame of predicted responses for each left out observation
  G = data.frame('Mrow' = rownames(M)[(n+1):N])
  
  # OLS for prediction function
  if(pred.fun=='lm'){
    OLS = lm(Ysample~., data=cbind(Ysample, Xsample))
    # beta-hat for entire sample
    betahat = OLS$coefficients 
    # beta-hat when dropping each individual observation
    betahatDropped = -1*lm.influence(OLS, do.coef=TRUE)$coefficients + matrix(rep(betahat, each=n), nrow=n)
    # f_i's
    M$f = c(rowSums(cbind(rep(1, n), Xsample) * betahatDropped), rep(NA, N-n))
    # G matrix: estimates when leaving out individual observations
    G = cbind(G, as.matrix(cbind(rep(1, N-n), Xother)) %*% t(betahatDropped))
    names(G) = c(names(G)[1], paste0('leftOutObs', names(G)[2:length(names(G))]))
  }
  
  # random forest for prediction function
  # using out-of-bag predictions/trees for in-sample, entire random forest for out-of-sample
  else if(pred.fun=='rf'){
    # random forest for entire sample
    f = randomForest(Ysample~., data=cbind(Ysample, Xsample))
    # predictions for sample -- out-of-bag by default
    M$f = c(f$predicted, rep(NA, N-n))
    # predictions for out-of-sample predictions
    # ignore exact predictions when sample obs. j is out-of-bag since estimates average out to be entire random forest
    g = predict(f, newdata=Xother)
    # G matrix: make entire row the prediction
    G = cbind(G, matrix(rep(g, n), nrow=N-n, ncol=n))
    names(G) = c(names(G)[1], paste0('leftOutObs', 1:n))
  }
  
  # sample mean for prediction function
  else if(pred.fun=='mean'){
    M$f = c(n/(n-1)*mean(Ysample) - 1/(n-1)*Ysample, rep(NA, N-n))
    G = cbind(G, matrix(rep(n/(n-1)*mean(Ysample) - 1/(n-1)*Ysample, each=N-n), nrow=N-n, ncol=n))
    names(G) = c(names(G)[1], paste0('leftOutObs', 1:n))
  }
  
  # create gamma matrix
  gammas = matrix(0, nrow=N-n, ncol=n)
  for(strat in unique(strats_samp)){
    gammas[strats_other==strat, strats_samp==strat] = 1/sum(strats_samp==strat)
  }
  
  # g_i (sum of f's * gammas) for observations not in s
  M$g = c(rep(NA, n), rowSums(G[,-which(names(G)=='Mrow')]*gammas))
  
  # h_i: f_i if i in sample, g_i if i not in sample
  M$h = ifelse(M$s==1, M$f, M$g)
  
  # pi_i: add P(i in S) to M
  M$pis = c(pis, rep(NA, N-n))
  
  # Y-hat formula
  M$Yhat = ifelse(M$s==1, M$h + 1/M$pis * (M$Y-M$h), M$h)
  
  muHat = mean(M$Yhat)
  
  # variance estimation
  tempMatrix = as.matrix(na.omit((M$Y-M$f))) %*% t(as.matrix(na.omit(M$Y-M$f)))
  varEst = 1/N^2 * with(M[M$s==1,], sum((1-pis)/pis^2*(Y-f)^2)) + 1/N^2 * sum((pijs*tempMatrix)[upper.tri(tempMatrix)])
  
  return(list('muHat' = muHat, 'M' = M, 'G' = G, 'varEst'=varEst))
}


summaryStats = function(ests, muHats, varEsts, yMean){
  # get bias, p-value, true variance, mean est. variance, and 95% CI coverage
  df = data.frame()
  for(n in unique(ests$n)){
    subset = ests[ests$n==n,]
    bias = yMean-mean(subset[[muHats]])
    trueVar = var(subset[[muHats]])
    pval = t.test(subset[[muHats]]-yMean)$p.value
    estVar = mean(subset[[varEsts]])
    lwrs = subset[[muHats]] - qt(.975, df=n-1) * sqrt(subset[[varEsts]])
    uprs = subset[[muHats]] + qt(.975, df=n-1) * sqrt(subset[[varEsts]])
    ciCov = sum(rowSums(cbind(yMean>lwrs, yMean<uprs))==2)/nrow(subset)
    temp = data.frame(n, bias, pval, trueVar, estVar, ciCov)
    df = rbind(df, temp)
  }
  return(df %>% round(., 4))
}


modelAssisted_OLS = function(Ysample, Xsample, XcolSums, N, probs, jointProbs){
  if(is.null(dim(Xsample))){ # X a vector, convert to matrix
    n = length(Xsample)
    Xsample = matrix(c(rep(1, n), Xsample), ncol=2)
  }
  else{ # add column of 1's
    n = nrow(Xsample)
    Xsample = cbind(rep(1, n), Xsample)
  }
  XcolSums = c(N-n, XcolSums)
  # full OLS using entire sample
  full_sample_df = rbind(Ysample, Xsample)
  OLS_full = lm(Ysample~as.matrix(Xsample)+0)
  resids = OLS_full$residuals
  muhat = 1/N * sum(resids/probs) + 1/N* (XcolSums + colSums(Xsample)) %*% OLS_full$coefficients
  pipj_1 = 1/(probs %o% probs); diag(pipj_1) = 0
  pij_1 = 1/jointProbs; diag(pij_1) = 0
  resids_cross = resids %o% resids; diag(resids_cross) = 0
  varEst = 1/N^2 * sum((1-probs)/probs^2 * resids^2) +
    1/N^2 * sum((pipj_1-pij_1) * resids_cross)
  return(list('muhat'=muhat, 'varEst'=varEst))
}

modelAssisted_ML = function(Ysample, Xsample, Xother, probs, jointProbs, method){
  N = nrow(Xsample) + nrow(Xother)
  # fit model-assisted estimator for specified machine learning algorithms
  if(method=='rf'){ # random forest
    fit_fun = randomForest(y=Ysample, x=Xsample)
  }
  if(method=='svm'){
    fit_fun = svm(y=Ysample, x=Xsample)
  }
  if(method=='xgboost'){
    Xall = data.matrix(rbind(Xsample, Xother)) 
    sample_xgb = xgb.DMatrix(data=data.matrix(Xsample), label=Ysample)
    Xsample = xgb.DMatrix(data=data.matrix(Xsample))
    Xother = xgb.DMatrix(data=data.matrix(Xother))
    fit_fun = xgboost(data=sample_xgb, objective = "reg:squarederror", nrounds=50, verbose=0)
    #xother_preds = predict(fit_fun, newdata=Xother)
    resids = Ysample - predict(fit_fun, newdata=Xsample)
    muhat = 1/N * sum(predict(fit_fun, Xall)) + 1/N * sum(resids/probs)
    pipj_1 = 1/(probs %o% probs); diag(pipj_1) = 0
    pij_1 = 1/jointProbs; diag(pij_1) = 0
    resids_cross = resids %o% resids; diag(resids_cross) = 0
    varEst = 1/N^2 * sum((1-probs)/probs^2 * resids^2) + 1/N^2 * sum((pipj_1-pij_1) * resids_cross)
    return(list('muhat'=muhat, 'varEst'=varEst))
  }
  if(method=='lasso'){
    cv_model = cv.glmnet(data.matrix(Xsample), Ysample, alpha=1)
    lambda_min = cv_model$lambda.min
    fit_fun = glmnet(data.matrix(Xsample), Ysample, alpha=1, lambda=lambda_min)
    resids = Ysample - predict(fit_fun, newx=data.matrix(Xsample))
    muhat = 1/N * sum(predict(fit_fun, newx=data.matrix(rbind(Xsample, Xother)))) + 1/N * sum(resids/probs)
    pipj_1 = 1/(probs %o% probs); diag(pipj_1) = 0
    pij_1 = 1/jointProbs; diag(pij_1) = 0
    resids_cross = c(resids) %o% c(resids); diag(resids_cross) = 0
    varEst = 1/N^2 * sum((1-probs)/probs^2 * resids^2) + 1/N^2 * sum((pipj_1-pij_1) * resids_cross)
    return(list('muhat'=muhat, 'varEst'=varEst))
  }
  if(method=='logreg1'){
    cv_model = cv.glmnet(data.matrix(Xsample), Ysample, alpha=1, family='binomial')
    lambda_min = cv_model$lambda.min
    fit_fun = glmnet(data.matrix(Xsample), Ysample, alpha=1, lambda=lambda_min, family='binomial')
    resids = Ysample - predict(fit_fun, newx=data.matrix(Xsample), type='response')
    muhat = 1/N * sum(predict(fit_fun, newx=data.matrix(rbind(Xsample, Xother)), type='response')) + 1/N * sum(resids/probs)
    pipj_1 = 1/(probs %o% probs); diag(pipj_1) = 0
    pij_1 = 1/jointProbs; diag(pij_1) = 0
    resids_cross = c(resids) %o% c(resids); diag(resids_cross) = 0
    varEst = 1/N^2 * sum((1-probs)/probs^2 * resids^2) + 1/N^2 * sum((pipj_1-pij_1) * resids_cross)
    return(list('muhat'=muhat, 'varEst'=varEst))
  }
  resids = Ysample - predict(fit_fun, newdata=Xsample)
  muhat = 1/N * sum(predict(fit_fun, rbind(Xsample, Xother))) + 1/N * sum(resids/probs)
  pipj_1 = 1/(probs %o% probs); diag(pipj_1) = 0
  pij_1 = 1/jointProbs; diag(pij_1) = 0
  resids_cross = resids %o% resids; diag(resids_cross) = 0
  varEst = 1/N^2 * sum((1-probs)/probs^2 * resids^2) + 1/N^2 * sum((pipj_1-pij_1) * resids_cross)
  return(list('muhat'=muhat, 'varEst'=varEst))
}

