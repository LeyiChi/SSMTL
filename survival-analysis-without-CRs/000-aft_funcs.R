
survfunc = function (object, t, newdata, name = "t") {
  newdata$ID_SurvivalCurves = 1:nrow(newdata)
  newdata <- do.call(rbind, rep(list(newdata), length(t)))
  t <- rep(t, each = nrow(newdata)/length(t))
  if (class(object) != "survreg") 
    stop("not a survreg object")
  lp <- predict(object, newdata = newdata, type = "lp")
  if (object$dist %in% c("weibull", "exponential")) {
    newdata$pdf <- dweibull(t, 1/object$scale, exp(lp))
    newdata$cdf <- ifelse(t == 0,0,
                          ifelse(is.nan(pweibull(t, 1/object$scale, exp(lp))),1,pweibull(t, 1/object$scale, exp(lp))))
    newdata$haz <- exp(dweibull(t, 1/object$scale, exp(lp), 
                                log = TRUE) - pweibull(t, 1/object$scale, exp(lp), 
                                                       lower.tail = FALSE, log.p = TRUE))
  }
  else if (object$dist == "lognormal") {
    newdata$pdf <- dlnorm(t, lp, object$scale)
    newdata$cdf <- plnorm(t, lp, object$scale)
    newdata$haz <- exp(dlnorm(t, lp, object$scale, log = TRUE) - 
                         plnorm(t, lp, object$scale, lower.tail = FALSE, log.p = TRUE))
  }
  else if (object$dist == "gaussian") {
    newdata$pdf <- dnorm(t, lp, object$scale)
    newdata$cdf <- pnorm(t, lp, object$scale)
    newdata$haz <- exp(dnorm(t, lp, object$scale, log = TRUE) - 
                         pnorm(t, lp, object$scale, lower.tail = FALSE, log.p = TRUE))
  }
  else if (object$dist == "loglogistic") {
    newdata$pdf <- dlogis(log(t), lp, object$scale)/t
    newdata$cdf <- plogis(log(t), lp, object$scale)
    newdata$haz <- exp(dlogis(log(t), lp, object$scale, log = TRUE) - 
                         log(t) - plogis(log(t), lp, object$scale, lower.tail = FALSE, 
                                         log.p = TRUE))
  }
  else if (object$dist == "logistic") {
    newdata$pdf <- dlogis(t, lp, object$scale)
    newdata$cdf <- plogis(t, lp, object$scale)
    newdata$haz <- exp(dlogis(t, lp, object$scale, log = TRUE) - 
                         dlogis(t, lp, object$scale, lower.tail = FALSE, log.p = TRUE))
  }
  else {
    stop("unknown distribution")
  }
  newdata$sur <- 1 - newdata$cdf
  newdata[name] <- t
  return(newdata)
}

AFT = function(training, testing, AFTDistribution){
  tryCatch({
    AFTMod = survreg(Surv(time, os)~race + sex + site + hist + grade +
                       ajcc7t + ajcc7n + ajcc7m + surgery + radiation + age,
                     data = training, dist = AFTDistribution)
    
    trainingTimes = sort(unique(training$time))
    if(0 %in% trainingTimes){
      timesToPredict = trainingTimes
    } else {
      timesToPredict = c(0,trainingTimes)
    }
    survivalCurves = survfunc(AFTMod, newdata = testing, t = timesToPredict)
    survivalCurvesTrain = survfunc(AFTMod, newdata = training, t = timesToPredict)
  },
  error = function(e) {
    message(e)
    warning("AFT failed to converge.")
  })
  if(!exists("AFTMod") | !exists("survivalCurves")){
    return(NA)
  }
  
  probabilities = survivalCurves$sur
  probabilitiesTrain = survivalCurvesTrain$sur

  probabilityMatrix = matrix(probabilities, ncol = nrow(testing),byrow = T)
  probabilityTrainMatrix = matrix(probabilitiesTrain, ncol = nrow(training),byrow = T)
  
  curvesToReturn = cbind.data.frame(time = timesToPredict, probabilityMatrix)
  timesAndCensTest = cbind.data.frame(time = testing$time, delta = testing$os)
  timesAndCensTrain = cbind.data.frame(time = training$time, delta = training$os)
  trainingCurvesToReturn = cbind.data.frame(time = timesToPredict, probabilityTrainMatrix)
  return(list(TestCurves = curvesToReturn, TestData = timesAndCensTest,TrainData = timesAndCensTrain,TrainCurves= trainingCurvesToReturn))  
}
